## evaluationmouvementsgmmPython
This is a Python3 version of 'EvaluationMouventsgmm'. The purpose is to allow the Poppy to perform mouvment learning and evaluation autonomously. Make sure to put the data files in the same directory. 

To start the program, use the commands below. 

```
python mainLearning.py
python mainEvaluation.py
```
Make sure that you have Python3 installed on your computer. If 'python' is not the corresponding environment variable for Python3, replace it with your environment variable for Python3.

'MainLearning.py' should create a file named 'model.txt', it stores the data of our trained GMM model in serialized format, it's normal that you can't read it directely. It will then be read in 'mainEvaluation.py'.




