import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import math
import operator
def getRelDocs(filename):
    '''Read file'''
    relDocs={}
    file=open(filename,'r')
    newline=file.readline() #First Query Document number should be here
    readingQuery=True;
    while(newline):
        queryNum="Q"+newline.split()[1]
        relDocs[queryNum]=[]
        newline=file.readline() #read the text after the document number
        while(readingQuery):
            if(":" in newline):
                relDocs[queryNum].append(newline.split('\n')[0].split()[0])
                newline=file.readline()
            if('-1' in newline):
               readingQuery=False;
               newline=newline.replace("-1",'')
            relDocs[queryNum]=relDocs[queryNum]+newline.split("\n")[0].split()
            newline=file.readline()
            if(newline=="\n"):
                newline=file.readline()
        readingQuery=True
    file.close()
    return relDocs
def getDoc2TermDict(PL):
    docDict={}
    for term in PL:
        for doc in PL[term][1]:
            if (doc not in docDict):
                docDict[doc]=[]
            docDict[doc].append(term)
    return (docDict)
    
def getCorpusPL(postlist,filename,stemmerOption,filterOption):
    '''Read file'''
    file=open(filename,'r')
    newline=file.readline() #First Document number should be here
    countDoc=0
    while(newline):
        countDoc=countDoc+1
        docNum=newline.split("\n")[0].split()[1]
        docText=""
        newline=file.readline() #read the text after the document number
        while(newline and "***************" not in newline):
            newline=newline.split('\n')[0]
            docText=docText+" "+newline
            newline=file.readline()
        docText=docText.lower() #easier to filter out when lowercase
        docWords=word_tokenize(docText) #Tokenize the terms
        if(filterOption=="yes"):
            stopWords=set(stopwords.words('english'))
            filteredWords=[]
            for word in docWords:
                if (str(word) not in stopWords):
                    filteredWords.append(word)
            docWords=filteredWords
        if(stemmerOption=="yes"):
            stemmer=PorterStemmer()
            for i in range(len(docWords)):
                docWords[i]=stemmer.stem(docWords[i])
		    
        for word in docWords:
            if(word not in postlist):
                postlist[word]=[0,{}]
            if(docNum not in postlist[word][1]):
                postlist[word][0]=postlist[word][0]+1
                postlist[word][1][docNum]= [0]
            postlist[word][1][docNum][0]= postlist[word][1][docNum][0]+1
        newline=file.readline() #new Doc begins if exists
        
    file.close()
    postlist["docSize"]=postlist["docSize"]+countDoc;
    return(postlist)

def getQueryInfo(filename,stemmerOption,filterOption):
    queryInfo={}
    '''Read file'''
    file=open(filename,'r')
    newline=file.readline() #First Query Document number should be here
    readingQuery=True;
    while(newline):
        queryNum="Q"+newline.split('\n')[0]
        queryInfo[queryNum]={}
        queryText=""
        newline=file.readline() #read the text after the document number
        while(readingQuery):
            if('#' in newline):
               readingQuery=False;
               newline=newline.replace("#",'')
            newline=newline.split('\n')[0]
            queryText=queryText+" "+newline
            newline=file.readline()
        queryText=queryText.lower() #easier to filter out when lowercase
        queryWords=word_tokenize(queryText) #Tokenize Terms
        if(filterOption=="yes"):
            stopWords=set(stopwords.words('english'))
            filteredWords=[]
            for word in queryWords:
                if (str(word) not in stopWords):
                    filteredWords.append(word)
            queryWords=filteredWords
        if(stemmerOption=="yes"):
            stemmer=PorterStemmer()
            for i in range(len(queryWords)):
                queryWords[i]=stemmer.stem(queryWords[i])
        for term in queryWords:
            if term not in queryInfo[queryNum]:
                queryInfo[queryNum][term]=[0]
            queryInfo[queryNum][term][0]=queryInfo[queryNum][term][0]+1
        readingQuery=True
    file.close()
    return queryInfo

def getWeights(PL,termWeightOption,docTotal):
    for Term in PL:
        df=PL[Term][0]
        idf=math.log(docTotal/df,10)
        for doc in PL[Term][1]:
            weight=PL[Term][1][doc][0]
            if(termWeightOption=="yes"):
                weight=weight*idf
            PL[Term][1][doc].append(weight)
    return PL
def getQueryWeights(Info,PL,termWeightOption,docTotal):
    for query in Info:
        for term in Info[query]:
            idf=0
            if(term in PL):
                df=PL[term][0]
                idf=math.log(docTotal/df,10)
            weight=Info[query][term][0] #tf
            if(termWeightOption=="yes"):
                weight=weight*idf
            Info[query][term].append(weight)
    return Info

            
def cosineScore(query,queryInfo,docTermDict,corpusPL,docCount):
    scores={}
    lengths={}
    for doc in range(docCount):
        if doc in docTermDict:
            addUp=0
            for term in docTermDict[doc]:
                addUp=addUp+corpusPL[term][1][doc][1]
                lengths[doc]=(addUp)**(1/2)
        else:
            lengths[doc]=0;
    for term in queryInfo[query]:
        weighttq=queryInfo[query][term][1]
        if(term in corpusPL):
            for docu in corpusPL[term][1]:
                if docu not in scores:
                    scores[docu]=0
                scores[docu]=scores[docu]+(corpusPL[term][1][docu][1]*weighttq)
    for doc in range(docCount):
            if doc in scores:
                scores[doc]=scores[doc]/lengths[doc]
            else:
                scores[doc]=0;
    return(scores)

def getSortedList(scoreDict):
    sortedList=sorted(scoreDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedList

def getKPerc(K,relDoc,sortedScoreDocs):
    returnInfo=[0,0]
    relTotal=0
    listPosDoc=[]
    for i in range (K):
        listPosDoc.append(sortedScoreDocs[i][0])
    for doc in listPosDoc:
        if (doc in relDoc[1:]):
            relTotal=relTotal+1
    returnInfo[0]=float(relTotal)/float(relDoc[0])
    returnInfo[1]=float(relTotal)/float(len(listPosDoc))
    return returnInfo

def main():
    '''Determine desired parameters'''
    stemmerOption=str(input("Would you like to use a Stemmer?: " )).lower()
    filterOption=str(input("Would you like to filter out stopwords?: ")).lower()
    termWeightOption=str(input("Would you like to use tf-idf for term weights?(term frequency otherwise): "))
    print("Preprocessing. Loading...")

    '''Parsing and creating Postlist of the corpus'''
    corpusPL={}
    corpusPL["docSize"]=0;
    ##Doc 1-500
    corpusPL=getCorpusPL(corpusPL,'LISA0.001',stemmerOption,filterOption)
    #Doc 501-1000
    corpusPL=getCorpusPL(corpusPL,'LISA0.501',stemmerOption,filterOption)
    #Doc 1001-1500
    corpusPL=getCorpusPL(corpusPL,'LISA1.001',stemmerOption,filterOption)
    #Doc 1501-2000
    corpusPL=getCorpusPL(corpusPL,'LISA1.501',stemmerOption,filterOption)
    #Doc 2001-2500
    corpusPL=getCorpusPL(corpusPL,'LISA2.001',stemmerOption,filterOption)
    #Doc 2501-3000
    corpusPL=getCorpusPL(corpusPL,'LISA2.501',stemmerOption,filterOption)
    #Doc 3001-3500
    corpusPL=getCorpusPL(corpusPL,'LISA3.001',stemmerOption,filterOption)
    #Doc 3501-4000
    corpusPL=getCorpusPL(corpusPL,'LISA3.501',stemmerOption,filterOption)
    #Doc 4001-4500
    corpusPL=getCorpusPL(corpusPL,'LISA4.001',stemmerOption,filterOption)
    #Doc 4501-5000
    corpusPL=getCorpusPL(corpusPL,'LISA4.501',stemmerOption,filterOption)
    #Doc 5001-5500
    corpusPL=getCorpusPL(corpusPL,'LISA5.001',stemmerOption,filterOption)
    #Doc 5501-5627
    corpusPL=getCorpusPL(corpusPL,'LISA5.501',stemmerOption,filterOption)
    #Doc 5628-5849
    corpusPL=getCorpusPL(corpusPL,'LISA5.627',stemmerOption,filterOption)
    #Doc 5850-6004
    corpusPL=getCorpusPL(corpusPL,'LISA5.850',stemmerOption,filterOption)
    docCount=corpusPL["docSize"]
    del corpusPL["docSize"]
    docTermDict=getDoc2TermDict(corpusPL)
    corpusPL=getWeights(corpusPL,termWeightOption,docCount)
    
    

    '''Paring and creating Postlist of the queries'''
    queryInfo=getQueryInfo('LISA.QUE',stemmerOption,filterOption)
    queryInfo=getQueryWeights(queryInfo,corpusPL,termWeightOption,docCount)
    queryRelDict=getRelDocs('LISA.REL')
    checkQuery=True
    query=str(input("Choose a query (max #=35) [Format Q#](q to quit): ")).upper()
    if(query=="Q"):
        checkQuery=False
    while(checkQuery):       
        scoreDict=cosineScore(query,queryInfo,docTermDict,corpusPL,docCount)
        sortedScoreDocs=getSortedList(scoreDict)
        firstK=int(input("Choose your K (max=6004): "))
        if("D" not in query):
            relDoc=queryRelDict[query] 
            secondK=int(input("Choose your second K  (max=6004): "))
            thirdK=int(input("Choose your third K  (max=6004): "))
            print("Top",firstK,": Recall:",getKPerc(firstK,relDoc,sortedScoreDocs)[0],", Precision: ",getKPerc(firstK,relDoc,sortedScoreDocs)[1])
            print("Top",secondK,": Recall:",getKPerc(secondK,relDoc,sortedScoreDocs)[0],", Precision: ",getKPerc(secondK,relDoc,sortedScoreDocs)[1])
            print("Top",thirdK,": Recall:",getKPerc(thirdK,relDoc,sortedScoreDocs)[0],", Precision: ",getKPerc(thirdK,relDoc,sortedScoreDocs)[1])
        if("D" in query):
            for i in range(firstK):
                 print(sortedScoreDocs[i][0])
            for i in range(docCount):
                if (sortedScoreDocs[i][0]==(query.split("D")[1])):
                    print("found at : ",i)
        query=str(input("Choose a query [Format Q#](q to quit): ")).upper()
        if(query=="Q"):
            checkQuery=False
         
main()

