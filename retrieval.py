"""BM25 plus corpus-derived identifier expansion. No embedding download."""
import collections
import json
import math
import re

STOP = set('a an the is are was were of on in to for from with what which when where how does do and or it its this that at by as be'.split())


def tokens(s):
    words = re.findall(r'[a-z0-9]+(?:[-_.#][a-z0-9]+)*#?',s.lower())
    out = []
    for w in words:
        if w not in STOP:
            out.append(w)
            if any(c in w for c in '-_.#'):
                out.extend(p for p in re.split(r'[-_.#]',w) if p and p not in STOP)
    return out


def identifiers(text):
    # Prefer specific ticket/part/code IDs; ignore generic prose and plain numbers.
    return set(re.findall(r'\b(?:[A-Z]{2,}[\-_][A-Z0-9]+(?:[\-_][A-Z0-9]+)*|[A-Z]{1,5}\d{2,})\b',text))


def chunks_for(source, parts, size=1500):
    result = []
    # Coalesce paragraphs so Word prose stays in context; table rows already have headers.
    combined, label = '', ''
    for part in parts:
        text = part.get('text','').strip()
        if not text:
            continue
        label = part.get('label','')
        if combined and len(combined)+len(text)>size:
            result.append({'source':source,'text':combined,'label':'document'})
            combined = ''
        if len(text)<=size:
            combined += ('\n' if combined else '') + f'[{label}]\n{text}'
            continue
        if combined:
            result.append({'source':source,'text':combined,'label':'document'})
            combined = ''
        start = 0
        while start < len(text):
            end = min(start+size,len(text))
            if end < len(text):
                boundary = text.rfind('\n',start+size//2,end)
                if boundary>start:
                    end=boundary
            result.append({'source':source,'text':text[start:end],'label':label})
            if end>=len(text):
                break
            start=max(start+1,end-180)
    if combined:
        result.append({'source':source,'text':combined,'label':label})
    return result


class Index:
    def __init__(self,chunks):
        self.chunks = [{**c,'id':i} for i,c in enumerate(chunks)]
        self.tfs = [collections.Counter(tokens(c['source']+' '+c['text'])) for c in self.chunks]
        self.df = collections.Counter(k for tf in self.tfs for k in tf)
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lengths)/max(1,len(chunks))
        self.links = collections.defaultdict(set)
        for i,c in enumerate(self.chunks):
            for ident in identifiers(c['text']):
                self.links[ident].add(i)

    def rank(self,query):
        scores = []
        n = len(self.chunks)
        for i,tf in enumerate(self.tfs):
            score=0.0
            for term in set(tokens(query)):
                freq=tf.get(term,0)
                if freq:
                    idf=math.log(1+(n-self.df[term]+0.5)/(self.df[term]+0.5))
                    score += idf*freq*2.2/(freq+1.2*(0.25+0.75*self.lengths[i]/max(1,self.avg)))
            scores.append(score)
        return scores

    def retrieve(self,query,limit=14):
        if not self.chunks:
            return []
        scores=self.rank(query)
        order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
        if not scores[order[0]]:
            return []
        seeds=[i for i in order[:8] if scores[i]>0]
        linked={}
        frontier=seeds[:4]
        for hop in range(2):
            next_frontier=[]
            for i in frontier:
                for ident in identifiers(self.chunks[i]['text']):
                    matches=self.links[ident]
                    if len(matches)>max(12,len(self.chunks)//5):
                        continue
                    for j in matches:
                        if j!=i and self.chunks[j]['source']!=self.chunks[i]['source']:
                            linked[j]=max(linked.get(j,0),scores[i]*0.7/(hop+1))
                            next_frontier.append(j)
            frontier=list(dict.fromkeys(next_frontier))[:8]
        # Reserve slots for seed matches and multi-hop bridge matches.
        picked=seeds[:5]
        for j in sorted(linked,key=lambda i:-(scores[i]+linked[i])):
            if j not in picked:
                picked.append(j)
            if len(picked)>=limit-3:
                break
        for j in order:
            if scores[j]>0 and j not in picked:
                picked.append(j)
            if len(picked)>=limit:
                break
        return [self.chunks[i] for i in picked[:limit]]


def parse_json(text):
    decoder=json.JSONDecoder()
    for i,c in enumerate(text):
        if c=='{':
            try:
                value,_=decoder.raw_decode(text[i:])
                if isinstance(value,dict):
                    return value
            except ValueError:
                pass
    return {}


def normalize(s):
    return re.sub(r'[\s\-._·]','',s).upper()


def value_in_quote(answer, quote):
    chars=normalize(answer)
    if not chars:
        return False
    flexible=r'[\s\-._·]*'.join(re.escape(c) for c in chars)
    # Prevent accepting 18 from 180 or a version prefix from a longer version.
    pattern=r'(?<![A-Z0-9])'+flexible+r'(?![A-Z0-9]|[._-][A-Z0-9])'
    return re.search(pattern,quote.upper()) is not None


def validate_answer(obj,selected):
    empty={'answer':'','citations':[]}
    answer=obj.get('answer')
    evidence=obj.get('evidence')
    if not isinstance(answer,str) or not answer.strip() or not isinstance(evidence,list):
        return empty
    if len(answer)>160:
        return empty
    mapping={c['id']:c for c in selected}
    used=[]
    quotes=[]
    for item in evidence:
        if not isinstance(item,dict) or type(item.get('chunk')) is not int:
            return empty
        chunk=mapping.get(item['chunk'])
        quote=item.get('quote')
        if not chunk or not isinstance(quote,str) or not quote.strip():
            return empty
        # Quote matching only normalizes whitespace, not punctuation or numbers.
        if ' '.join(quote.split()) not in ' '.join(chunk['text'].split()):
            return empty
        used.append(chunk['source'])
        quotes.append(quote)
    if not used:
        return empty
    # Reject generated values absent from the cited evidence. This is a guard,
    # not a proof of semantic entailment or correct citation necessity.
    if not any(value_in_quote(answer,q) for q in quotes):
        return empty
    return {'answer':answer.strip(),'citations':sorted(set(used))}
