## Methods to test the correlation between categorical variables or between categorical and numerical variables:

Significance tests:

    Continuous vs. Nominal: run an ANOVA.
    Nominal vs. Nominal: run a chi-squared test.

Effect size (strength of association):

    Continuous vs. Nominal: calculate the intraclass correlation. 
    Nominal vs. Nominal: calculate Cramer's V. 
__________

Between categorical variables, use **Chi-square Analysis**:  

If we assume that two variables are independent, the values of the contingency table for those variables should be evenly distributed. Then we check the distance between the actual value and the uniformity.  
Null Hypothesis H0: variables are independent(uniform distribution).
If p value < 0.05, it means the two variables are not independent (correlated).  

Between categorical and numerical variables, use **ANOVA Analysis**:   

We calculated in-group variance and intra-class intra-group variance, and then compared them.  
If p value < 0.05, it means the two variables are not independent (correlated).

连续vs连续：相关分析
连续vs分类：T检验，方差检验
分类vs分类：卡方检验
