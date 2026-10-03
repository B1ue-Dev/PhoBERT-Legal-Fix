import sys
import os

# Windows PowerShell can otherwise default to a legacy code page which cannot
# print PAP_NER's Vietnamese label ``ĐT``.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

from vphoberttagger import LOGGER, Trainer, Predictor

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process...")
        Trainer.train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process...")
        Trainer.test()
    elif sys.argv[1] == 'predict':
        LOGGER.info("Start PREDICT process...")
        Predictor.tagging()
    elif sys.argv[1] == 'demo':
        LOGGER.info("Start PREDICT process...")
        comd = "PYTHONPATH=./ streamlit run vphoberttagger/demo.py -- " + " ".join(sys.argv[1:])
        print(comd)
        os.system(comd)

