"""
Created on Mon Jul  4 17:02:13 2022

Created by: Seyed Mohammad Kazempour
-------------------------------------------------------------------------------
File Description Here.
"""
import pandas as pd
import statsmodels.api as sm
import linearmodels

# generic_ols is a wrapper for regression results across all packages - I will 
# update this class with different models over time
class generic_ols:
    def __init__(self, est):
        
        if isinstance(est, sm.regression.linear_model.RegressionResultsWrapper):
            self.params = est.params
            self.pvalues = est.pvalues
            self.tvalues = est.tvalues
            self.bse = est.bse
            self.rsquared = est.rsquared
            self.nobs = est.nobs
            
        elif isinstance(est, linearmodels.panel.results.PanelEffectsResults):
            self.params = est.params
            self.pvalues = est.pvalues
            self.tvalues = est.tstats
            self.bse = est.std_errors
            self.rsquared = est.rsquared
            self.nobs = est.nobs

        elif isinstance(est, linearmodels.iv.results.AbsorbingLSResults):
            self.params = est.params
            self.pvalues = est.pvalues
            self.tvalues = est.tstats
            self.bse = est.std_errors
            self.rsquared = est.rsquared
            self.nobs = est.nobs
        
class linear_regression:
    def __init__(self, l_est):
        self.l_est = [generic_ols(est) for est in l_est]
        self.num_est = len(l_est)
        l_var = []
        for est in l_est:
            l_var = l_var + list(est.params.index)
        self.l_var = list(set(l_var))
        self.significant_digits = 2
        self.d_var_names = None
        self.columns = None
        self.caption = ''
        self.alignment = 'l' + 'c'*(self.num_est + 1)
        self.significance_levels = [0.10, 0.05, 0.01]
        self.aux_stat = 'se'
        self.tbl = pd.DataFrame()
        self.label = ''
        self.float_format = ".2f"
        self.extra_rows = None
        
    def set_var_list(self, l_var):
        self.l_var = list(set(l_var))
        
    def drop_vars(self, l_var):
        self.l_var = list(set(self.l_var) - set(l_var))
    
    def set_float_format(self, float_format):
        self.float_format = float_format
    
    def rename_columns(self, l_col_names):
        self.columns = l_col_names
        
    def significance_levels(self, sl):
        self.significance_levels = sl
        
    def rename_variables(self, d):
        self.d_var_names = d
        
    def fix_stats(self, b, p, s):
        b_str = format(b, self.float_format)
        if p > self.significance_levels[0]:
            pass
        elif p > self.significance_levels[1]:
            b_str = b_str + 'ONESTAR'
        elif p > self.significance_levels[2]:
            b_str = b_str + 'TWOSTAR'
        else:
            b_str = b_str + 'THREESTAR'
        s_str = format(s, self.float_format)
        s_str = f'({s_str})'
        return b_str, s_str
    
    def render(self, R2=False, obs=True, midrule=False, midrule_pad=False, 
               replace=None):
        index = pd.MultiIndex.from_product([self.l_var, ['', ' ']])
        columns = [f'({i})' for i in range(1, self.num_est + 1)] # to be renames later
        tbl = pd.DataFrame(dtype=str, index=index, columns=columns)
        
        for i, est in enumerate(self.l_est):
            
            # get the parameters
            b = est.params
            
            # get the auxiliary stat
            if self.aux_stat == 'p':
                s = est.pvalues
            elif self.aux_stat == 't':
                s = est.tvalues
            elif self.aux_stat == 'se':
                s = est.bse
            
            # get the p-values
            p = est.pvalues
            
            for v in self.l_var:
                if v in b.index:
                    bb, ss = self.fix_stats(b[v], p[v], s[v])
                    tbl.loc[(v, ''), columns[i]] = bb
                    tbl.loc[(v, ' '), columns[i]] = ss
                    
        # rename the rows if needed
        if self.d_var_names is not None:                        
            tbl = tbl.reindex(self.d_var_names.keys(), level=0)
            tbl = tbl.rename(index=self.d_var_names, level=0)
        
        # separate the extra rows by an empty row 
        if midrule_pad:
            tbl.loc['',:] = ''
        
        # add the mid-rule placeholder if needed
        if midrule:
            tbl.loc['', columns[-1]] = "ADDMIDRULEHERE"
            
        # add extra rows if indicated. extra_rows should be a dictionary with the label of the rows
        # as keys and the value of all columns passed as the value of the dictionary
        if self.extra_rows is not None:
            for r in self.extra_rows.keys():
                tbl.loc[r, :] = self.extra_rows[r]
        
        # add the R2 if indicated
        if R2:
            for i, est in enumerate(self.l_est):
                tbl.loc['R2', columns[i]] = format(est.rsquared, self.float_format)
        
        # add the observations if indicated
        if obs:
            for i, est in enumerate(self.l_est):
                tbl.loc['N', columns[i]] = format(int(est.nobs), ",")

        # rename the columns if needed
        if self.columns is not None:
            tbl.columns = self.columns
        
        # fix the missing cells
        tbl = tbl.fillna('')

        # save the table                
        self.tbl = tbl.copy()
        
        ### convert to latex
        tbl = tbl.style.to_latex(caption=self.caption, 
                                            label=self.label, 
                                            column_format=self.alignment,
                                            multirow_align='naive',
                                            hrules=True,
                                            clines=None)
        
        # custom replacement of strings
        if replace is not None:
            for item in replace.keys():
                tbl = tbl.replace(item, replace[item])
        
        # # replace the interaction operator *
        # tbl = tbl.replace('*','$\\times$')
        
        # replace the placeholders
        tbl = tbl.replace('ONESTAR','$^{*}$')
        tbl = tbl.replace('TWOSTAR','$^{**}$')
        tbl = tbl.replace('THREESTAR','$^{***}$')
        tbl = tbl.replace("ADDMIDRULEHERE \\\\", " \\\\ \\midrule")
        
        # save the string if needed in future
        self.latex_string = tbl
        
        # finished!
        return tbl
    
    def save(self, file_address):
        with open(file_address, 'w') as file:
            file.write(self.latex_string)
    
            
class summary_stat:
    def __init__(self, data, variables=None, pctiles=None):
        self.data = data
        self.variables = variables
        if self.variables is not None:
            self.num_vars = len(variables)
        else:
            self.num_vars = len(data.columns)
        self.num_stats = 3 + len(pctiles)
        self.pctiles = pctiles
        self.float_format = ".2f"
        self.d_var_names = None
        self.interpolation = 'nearest'
        self.d_var_types = dict(zip(variables, [float]*self.num_vars)) # by default all variables are float
        self.label=''
        self.caption=''
    
    def set_var_types(self, d_var_types):
        self.d_var_types = d_var_types
        
    def rename_vars(self, d_var_names):
        self.d_var_names = d_var_names
    
    def render(self):
        
        # get the relevant columns only
        if self.variables is not None:
            df = self.data[self.variables]
        else:
            df = self.data
        
        # arrange the table
        tbl = pd.DataFrame(dtype=str, index=self.variables, columns=['N', 'Mean', 'SD'] + ['P' + str(int(100*p)) for p in self.pctiles])
        
        # fill in the stats one by one
        for v in self.variables:        
            tbl.loc[v, 'N'] = format(int(df[v].count()), ",")
            tbl.loc[v, 'Mean'] = format(df[v].mean(), self.float_format)
            tbl.loc[v, 'SD'] = format(df[v].std(), self.float_format)
            
            for p in self.pctiles:
                if self.d_var_types[v] == float:
                    tbl.loc[v, "P"+str(int(100*p))] = format(df[v].quantile(p, interpolation=self.interpolation), self.float_format)
                elif self.d_var_types[v] == int:
                    tbl.loc[v, "P"+str(int(100*p))] = round(df[v].quantile(p, interpolation=self.interpolation))
        
        # reorder and rename the variables
        if self.d_var_names is not None:
            tbl = tbl.reindex(self.d_var_names.keys())
            tbl = tbl.rename(index=self.d_var_names)

        s = tbl.to_latex(column_format='l'+'c'*self.num_stats, label=self.label, caption=self.caption)
        self.tbl = tbl
        self.latex_string = s

        return s, tbl
    
    def save(self, file_address):
        with open(file_address, 'w') as file:
            file.write(self.latex_string)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        