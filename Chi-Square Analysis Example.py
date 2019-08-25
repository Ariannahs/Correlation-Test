import scipy.stats as stats
from scipy.stats import chi2_contingency 

## Feature Selection    

testColumns = ['label', 'ver', 'apptype', 'ip', 'city', 'province', 'reqrealip',
       'dvctype', 'make', 'ntt', 'carrier', 'orientation', 'lan', 'h_w', 'ppi',
       'hour']
       
## Introduce ChiSquare Class     

class ChiSquare:
    def __init__(self, dataframe):
        
		self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None  #Chi Test Statistic
        self.dof = None
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} - IMPORTANT predictor. Chi-Statistic:{1}, P-value:{2}".format(colX,self.chi2,self.p) 
        else:
            result="{0} - NOT important predictor. Chi-Statistic:{1}, P-value:{2}".format(colX,self.chi2,self.p)

        print(result)
        
    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved =  pd.crosstab(Y,X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)
        
        
## Initialize ChiSquare Class   
   
cT = ChiSquare(train_clean)

for var in testColumns:
    cT.TestIndependence(colX=var, colY='label')
