# KoBERT-NER

- forked from monologg/KoBERT-NER
- pii detection using NER with KoELECTRA

## Dependencies

- torch==1.7.0
- transformers==3.31.0
- seqeval>=0.0.12

## Dataset

- **AI Hub**의 민원(콜센터) 질의-응답 Dataset 사용 ([link](https://aihub.or.kr/aidata/30716))
- **국립국어원 모두의 말뭉치**의 개체명 분석 말뭉치 사용 ([link](https://corpus.korean.go.kr/main.do))
- 이미 비식별화 처리된 부분은 임의로 데이터 생성하여 채워 넣었음
- token 단위로 라벨링
  - **2022-04-12** Train (5,442) / Test (1,361) 

## History

- main.py, data_loader.py: token 단위로 라벨링한 파일을 읽어들이도록 수정
- predict.py: token 단위로 predict한 결과 출력하도록 수정


## Usage

```bash
$ python3 main.py --data_dir ./res --model_type koelectra-base --do_train --do_eval --train_batch_size 64 --eval_batch_size 64 --logging_steps 71 --save_steps 15 --num_train_epochs 15
```

- `--write_pred` 옵션을 주면 **evaluation의 prediction 결과**가 `preds` 폴더에 저장됩니다.

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results


