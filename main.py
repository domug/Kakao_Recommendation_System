from helper import *
import warnings
warnings.filterwarnings(action='ignore')  # 경고 메시지 무시
import time

############################################################################################
print("###########################################################################################################################################################################")
print("###########################################################################################################################################################################")
print("환경 설정 중입니다 잠시만 기다려주세요..")
start_time = time.time()

# 곡 메타 데이터
song_meta = pd.read_json("./data/song_meta.json")

# 플레이리스트 데이터
plylst = pd.read_json("./data/train.json")   # 플레이리스트
plylst["nid"] = range(len(plylst))           # 곡 ID 설정
plylst["embedding_vector"] = plylst.tags.apply(get_embedding_vector)   # 태그들의 임베딩 벡터 추가

# 플레이리스트 데이터 전처리
n_songs, song_id_sid, song_sid_id, plylst_use, songs_spr = get_plylst(plylst)

print("환경 설정이 완료되었습니다.")

print("###########################################################################################################################################################################")
print("###########################################################################################################################################################################")
print()

run_recommendation = "y"
while run_recommendation != "n":

    # 유저에게 키워드 입력받기
    stop = True
    keyword_list = []
    print("안녕하세요 사용자님, 오늘은 어떤 노래를 듣고 싶으신가요?\n사용자님의 취향이나 기분을 표현할 수 있는 단어를 마음껏 입력해주세요.(q를 입력하면 종료됩니다)\n")
    while stop:
        user_input = input("입력:").lower()
        if user_input == "q":
            stop = False
            continue
        if user_input in keyword_list:
            print("앗, 해당 단어는 이미 입력하신 것 같아요. 다시 입력해주세요.")
            print()
            continue
        if user_input == "":
            print("오류입니다. 다시 입력해주세요.")
            continue
        keyword_list.append(user_input)
        print()
    print()
    print()

    print("입력해주신 단어들은 {} 이군요?\n사용자님께서 마음에 들어하실 곡들을 추천해드리는 중이에요. 잠시만 기다려주세요.\n\n".format(keyword_list))

    # 추천을 위해 코사인 유사도 계산
    try:
        subset_plylst = get_plylst_cs(keyword_list, plylst)
    except:
        print("입력해주신 단어들만으로는 추천을 해드리기가 어려워요 ㅠㅠ 좀 더 구체적인 단어를 입력해주세요.")
        print("###########################################################################################################################################################################")
        print("###########################################################################################################################################################################")

        continue


    # 곡 추천
    recs = recommendation(subset_plylst, plylst_use, n_songs, song_sid_id, songs_spr)
    recs_df = get_songs(recs, song_meta)

    for i in range(len(recs_df)):
        issue_date = recs_df.iloc[i, 1]
        album_name = recs_df.iloc[i, 2]
        song_name = recs_df.iloc[i, 5]
        artist_name = recs_df.iloc[i, 7]

        print("<{}>\n앨범 제목: {}\n발매일자: {}\n아티스트 명: {}\n곡 제목:{}\n".format(i+1,
                                                                          album_name,
                                                                          issue_date,
                                                                          artist_name,
                                                                          song_name))

    print("계속 하시려면 y를, 종료하시려면 n을 입력해주세요.")
    run_recommendation = input("입력:")
    print("###########################################################################################################################################################################")
    print("###########################################################################################################################################################################")


print()
print("프로그램을 종료합니다.")



print("--- %s seconds ---" % (time.time() - start_time))

