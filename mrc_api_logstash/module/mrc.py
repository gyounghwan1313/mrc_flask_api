
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MRC(object):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad")

    def convert_to_token(self,
                         question: str,
                         context: str) -> [list, list, list, list, list, list]:
        """
        질문, 본문를 토큰으로 변환
        koelectra모델의 최대 토큰 수는 512

        ## Model Input
        token_id: "[CLS] 질문 [SEP] 본문 [SEP]"의 토큰 형태로 모델에 input
                    토큰의 길이가 512보다 작은 경우, padding [0] 으로 512로 자릿수를 채움
        attention_mask : token_id는 1, padding은 0으로 변환하여 모델에 input
        token_type_id : "본문 + [SEP]"만 1로, 나머지는 0로 변환하여 모델에 input

        token_text : 토큰화한 "[CLS] 질문 [SEP] 본문 [SEP]" => 정답 텍스트를 추출하기 위해 사용

        :param question: 질문 text
        :param context: 본문 text
        :return: token_id, attention_mask, token_type_id, token_text
        """
        _token_question = self.tokenizer.tokenize('[CLS] ' + question + ' [SEP]')
        _token_question_id = self.tokenizer.convert_tokens_to_ids(_token_question)

        _token_context = self.tokenizer.tokenize(context + ' [SEP]')
        _token_context_id = self.tokenizer.convert_tokens_to_ids(_token_context)

        token_text = _token_question + _token_context
        token_id = _token_question_id + _token_context_id

        _pad_len = 512 - len(token_id)

        token_type_id = [0] * len(_token_question_id) + [1] * len(_token_context_id) + [0] * _pad_len
        attention_mask = [1] * len(token_id) + [0] * _pad_len
        token_id += [0] * _pad_len

        return token_id, attention_mask, token_type_id, token_text

    def predict_answer(self,
                       token_id: list,
                       attention_mask: list,
                       token_type_id: list,
                       token_text: list) -> [str, float]:

        with torch.no_grad():
            _start_logit, _end_logit = self.model(torch.tensor([token_id]).to("cpu"),
                                                  torch.tensor([attention_mask]).to("cpu"),
                                                  torch.tensor([token_type_id]).to("cpu"))

        _max_start = np.argmax(_start_logit)
        _max_end = np.argmax(_end_logit)

        _start_prob = sigmoid(_start_logit.max())
        _end_prob = sigmoid(_end_logit.max())

        mean_prob = (_start_prob + _end_prob) / 2

        if _max_start > _max_end:
            answer_text = "정답 없음"
        else:
            answer_text = self.tokenizer.convert_tokens_to_string(token_text[_max_start:_max_end + 1])

        return answer_text, round(float(mean_prob), 3) if float(mean_prob) > 0 else 0


if __name__ == '__main__':

    question = "1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의로 지명수배된 사람의 이름은?"
    context = """1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다. 1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. 같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. 경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, 12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다. 1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. 임종석은 청량리경찰서에서 약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다."""

    predict_model = MRC()
    token_id, attention_mask, token_type_id, token_text = predict_model.convert_to_token(question=question, context=context)
    answer, prob = predict_model.predict_answer(token_id, attention_mask, token_type_id, token_text)

    print("Answer :", answer)
    print("Probability :", prob)