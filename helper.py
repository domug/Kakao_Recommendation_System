from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pandas as pd
import random
import scipy.sparse as spr



###############################################################################################
## 플레이리스트 데이터 전처리

def get_plylst(plylst):

    ##### 곡 ID 인코딩 #####
    plylst_song = plylst['songs']
    song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
    song_dict = {x: song_counter[x] for x in song_counter} # 곡 ID: 빈도 수
    n_songs = len(song_dict)

    song_id_sid = dict()
    song_sid_id = dict()

    for i, t in enumerate(song_dict): # key 인덱스, key(곡 ID)
      song_id_sid[t] = i # id: index(sid)
      song_sid_id[i] = t # index(sid): id

    # 데이터프레임에 추가
    plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x
                                                        if song_id_sid.get(s) != None])

    plylst_use = plylst[['nid','updt_date','songs_id']] # 필요한 정보만 추출
    plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
    plylst_use = plylst_use.set_index('nid') # 새로운 플레이리스트 아이디로 인덱스 설정


    # 플레이리스트에 수록된 곡 별로 BM25 스코어를 계산해 리스트 형태로 추가
    n_plylst = plylst_use.shape[0]
    docCount_val = n_plylst
    avgFieldLength_song = plylst_use.num_songs.sum() / n_plylst
    plylst_use['bm25_song'] = plylst_use['songs_id'].map(lambda song_lst: bm25(song_lst,
                                                                               docCount_val,
                                                                               avgFieldLength_song,
                                                                               fieldLength=len(song_lst),
                                                                               song_counter=song_counter,
                                                                               song_sid_id=song_sid_id))

    # 곡 희소행렬
    row = np.repeat(range(len(plylst_use)), plylst_use['num_songs'])  # 플레이리스트의 nid를 각 곡의 개수만큼 반복
    col = [song for songs in plylst_use['songs_id'] for song in songs]  # 각 플레이리스트의 곡의 sid를 열로 늘어놓음
    dat = [value for values in plylst_use['bm25_song'] for value in values]  # bm25 value
    train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(len(plylst_use), n_songs))

    return n_songs, song_id_sid, song_sid_id, plylst_use, train_songs_A



###############################################################################################
## 곡 가중치 계산

def idf_song(docFreq_lst, docCount, song_counter, song_sid_id):
    """
    곡의 idf 계산 함수
    docCount: 전체 플레이리스트의 개수
    docFreq: 전체 플레이리스트에서 해당 곡/태그가 등장한 횟수
    """
    docFreq = np.array( [song_counter[song_sid_id[sid]] for sid in docFreq_lst] )
    return np.log(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5))


def tfNorm(termFreq, avgFieldLength, fieldLength, k1=1.2, b=0.75):
    """
    termFreq: 해당 플레이리스트에서 해당 곡/태그가 등장한 횟수(무조건 1일듯)
    k1: elasticsearch default는 1.2. 보통 1.2 혹은 2.0을 사용합니다. (가중치)
    b: elasticsearch default는 0.75 (가중치)
    avgFieldLength: 평균 플레이리스트의 길이(곡/태그의 개수)
    fieldLength: 해당 플레이리스트의 길이(곡/태그의 개수)
    """
    return (termFreq * (k1 + 1)) / (termFreq + k1 * (1 - b + b * fieldLength / avgFieldLength))


def bm25(docFreq_lst, docCount, avgFieldLength, fieldLength,
         song_counter, song_sid_id, k1=1.2, b=0.75, termFreq=1):
    return idf_song(docFreq_lst, docCount, song_counter, song_sid_id) * tfNorm(termFreq, avgFieldLength, fieldLength, k1=k1, b=b)


###############################################################################################
## 키워드 임베딩

word2vec_model = Word2Vec.load("./data/64features_30minwords_5context")

def get_embedding_vector(words, model=word2vec_model, num_features=64):

  # 출력 벡터 초기화
  feature_vector = np.zeros(num_features, dtype=np.float32)

  num_words = 0
  index2word_set = set(list(model.wv.index_to_key))   # 어휘사전

  # 어휘사전에 포함된 단어 벡터들의 평균값을 해당 텍스트의 임베딩 벡터로 사용
  for w in words:
    if w in index2word_set:
      num_words += 1
      feature_vector = np.add(feature_vector, model.wv[w])

  feature_vector = np.divide(feature_vector, num_words)
  return feature_vector.reshape(1, -1)



###############################################################################################
## 추천 시스템

def get_plylst_cs(input, plylst):
    """
    서브셋 플레이리스트들과 유저의 입력값 사이 코사인 유사도 계산해 데이터프레임에 추가
    """
    input_vec = get_embedding_vector(input)

    cs = []
    subset_plylst = plylst.sample(frac=0.15)
    for vec in subset_plylst["embedding_vector"]:
        if pd.Series(vec.sum()).isnull()[0]:  # 벡터 값이 없는 경우
            cs.append(0)
        else:
            cs.append(cosine_similarity(input_vec, vec)[0][0])


    subset_plylst["cosine_similarity"] = cs

    return subset_plylst


def get_songs(song_sid, song_meta):
    """
    곡 정보 반환
    """
    return song_meta[song_meta.id.isin(song_sid)]



def recommendation(subset_plylst, plylst_use, n_songs, song_sid_id, sparse_matrix):
    """
    곡 추천시스템
    """

    # 코사인 유사도 기준 상위 10개 플레이리스트의 곡을 저장
    rec_song_idx = []
    for pid in subset_plylst.sort_values("cosine_similarity", ascending=False).head(5).index:
        res = []
        p = np.zeros((n_songs, 1))
        p[plylst_use.loc[pid, 'songs_id']] = np.array(plylst_use.loc[pid, 'bm25_song']).reshape(-1, 1)  # 곡의 bm25 가중치로 대체

        # 기존의 플레이리스트와 해당 플레이리스트(pid) 간의 유사성 계산
        val = sparse_matrix.dot(p).reshape(-1)

        songs_already = plylst_use.loc[pid, "songs_id"]  # 해당 플레이리스트에 들어있는 곡의 id

        # 해당 플레이리스트의 곡들과 유사성 가중치를 곱해 곡의 추천 강도를 계산함
        cand_song = sparse_matrix.T.dot(val)

        # 추천 강도 내림차순 정렬 -> 상위 200개 추출
        # 150개 추출에서 수정함_중복곡을 빼면 100개보다 곡이 모자라는 현상 발생
        # cand_song_idx = cand_song.reshape(-1).argsort()[-150:][::-1]
        cand_song_idx = cand_song.reshape(-1).argsort()[-100:][::-1]

        # 해당 플레이리스트에 이미 있는 곡은 제외하고, 곡을 10개 추천
        cand_song_idx = pd.Series(cand_song_idx[np.isin(cand_song_idx, songs_already) == False]).sample(5)
        rec_song_idx_temp = [song_sid_id[i] for i in cand_song_idx]

        rec_song_idx = rec_song_idx + rec_song_idx_temp

    # 중복곡 삭제
    rec_song_idx = set(rec_song_idx)

    # 랜덤하게 10곡 샘플링
    rec_song_idx = random.sample(list(rec_song_idx), 10)

    return rec_song_idx
