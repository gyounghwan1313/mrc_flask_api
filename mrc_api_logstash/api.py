
import os
import json
import traceback
import socket
import datetime

import logging
import logging.handlers
import logstash

from flask import Flask, request, Response
from flask_restful import Api
from flask.views import MethodView



with open("./connection.json","r") as file:
    access_info = json.load(file)

logstash_host = access_info['AWS-logstash']['host']
logstash_port = access_info['AWS-logstash']['port']

logger = logging.getLogger('elk-mrc-predict')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s  : %(message)s')
stash = logstash.TCPLogstashHandler(logstash_host, logstash_port, version=1)
stash.setFormatter(formatter)
logger.addHandler(stash)


from module.mrc import MRC


## 인스턴스 생성
app = Flask(__name__)
api = Api(app)
model = MRC()

# 한글 설정
app.config['JSON_AS_ASCII'] = False

mandatory_key = ['question', 'context']

class Main(MethodView):

    def get(self):
        return "Hello World"

    def post(self):
        data = request.get_json()

        logger.info(f"==Get data : {data}")
        output = {"status": "OK",
                  "error_code": "",
                  "error_message": "",
                  "data": ""}

        try:
            for m_key in mandatory_key:
                if m_key not in data.keys():
                    output['status'] = "ERROR"
                    output['error_code'] = "ERROR-01"
                    output['error_code'] = f"{m_key} 키 없음"
                    raise Exception("get data error")

                if len(str(data[m_key]).strip()) == 0 or data[m_key] is None:
                    output['status'] = "ERROR"
                    output['error_code'] = "ERROR-02"
                    output['error_code'] = f"{m_key} 값 없음"
                    raise Exception("get data error")

            token_id, attention_mask, token_type_id, token_text = model.convert_to_token(question=data["question"], context=data["context"])
            answer, prob = model.predict_answer(token_id, attention_mask, token_type_id, token_text)

            output_data = {}
            output_data["question"] = data["question"]
            output_data["context"] = data["context"]
            output_data["answer"] = answer
            output_data["probability"] = prob
            output['data'] = output_data

            # return Response(response=json.dumps(output, ensure_ascii=False), mimetype='application/json')


        except Exception as e:
            logger.info(f'==ERROR_DEBUG : {traceback.format_exc()}')

            if output['error_code'] == "":
                output['status'] = "ERROR"
                output['error_code'] = "ERROR-99"
                output['error_message'] = f"{datetime.datetime.today()}-UNKNOWN"

            logger.info(f"==ERROR_RETURN : {output}, \n {data}")
        logger.info(f"==Output data : {output}")
        return Response(response=json.dumps(output, ensure_ascii=False), mimetype='application/json')


api.add_resource(Main, '/mrc')

if __name__ == '__main__':
    port = 8000
    print('Start Server.... port=',port)
    app.run(host='0.0.0.0', port=port, debug=True)