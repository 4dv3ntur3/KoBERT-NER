# 문장 분리 기능도 지원합니다.

from kiwipiepy import Kiwi
import time


start = time.time()
kiwi = Kiwi()
print(time.time()-start)


'''
text, start, end
'''


long_sample = "이 조례의 제정안에 대하여 의견이 있는 단체 또는 개인은 2022년 6월 22일까지 다음 사항을 기재한 의견을 작성하여 서대문구청장(참조 : 여성가족과, 주소 : 서울특별시 서대문구 연희로 248, 2층)에게 제출하여 주시기 바라며, 그 밖에 자세한 사항은 여성가족과(전화 : (02) 330-1689, FAX : (02) 330-1624, E-mail: egayeong@sdm.go.kr)로 문의하시기 바랍니다."

short_text = "포스코청암재단은 6일 오후 서울 강남구 포스코센터에서 '2022 포스코청암상' 시상식을 열어 수상자 4명에게 각각 상패와 상금 2억 원을 수여했다."




start = time.time()
a = kiwi.split_into_sents(short_text)
print(time.time()-start)

print(a)


start = time.time()
b = kiwi.split_into_sents(long_sample)
print(time.time()-start)

print(b)


start = time.time()
a = kiwi.split_into_sents(short_text)
print(time.time()-start)

print(a)



# for line in a:
#     print(line.text)

# print("문장 개수: ", len(a))

