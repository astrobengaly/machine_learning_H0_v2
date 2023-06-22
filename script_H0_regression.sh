# !/bin/sh
# for other nsim values, just change 10 to other numbers (supports a maximum value of nsims=100)
# for other sigh/h values, just change 0.03 to other numbers (supports sigh/h=0.008, 0.01, 0.03, 0.05, 0.08)
# python H0_regression.py 10 0.03 20 EXT
# python H0_regression.py 10 0.03 30 EXT
# python H0_regression.py 10 0.03 50 EXT
# python H0_regression.py 10 0.03 80 EXT

python H0_regression.py 10 0.03 20 ANN
python H0_regression.py 10 0.03 30 ANN
python H0_regression.py 10 0.03 50 ANN
python H0_regression.py 10 0.03 80 ANN

# python H0_regression.py 10 0.03 20 GBR
# python H0_regression.py 10 0.03 30 GBR
# python H0_regression.py 10 0.03 50 GBR
# python H0_regression.py 10 0.03 80 GBR
#
# python H0_regression.py 10 0.03 20 SVM
# python H0_regression.py 10 0.03 30 SVM
# python H0_regression.py 10 0.03 50 SVM
# python H0_regression.py 10 0.03 80 SVM
