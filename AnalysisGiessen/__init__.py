import numpy  as np 
import pandas as pd
from   scipy.ndimage import gaussian_filter1d
from   scipy.signal  import find_peaks

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os


class analyseGiessen:
    def __init__(self, file=None, df=None, t_resolution=None):
        assert not ((file is None) and (df is None)) , "Either file or df need to be nonzero"
        
        if file is not None:
            self._file = os.path.join('CohortDataRaw', file)
            self._date = self._file.split('/')[2]
            
            self._df = pd.read_csv(self._file, on_bad_lines='skip', index_col='Zeit')
            self._df.index = self._df.apply(lambda line : self._date + ' ' +  line.name, axis=1)
            self._df.index = pd.to_datetime(self._df.index, format='%Y-%m-%d %H:%M:%S:%f')
            
            self._df['Pressure'] = self._df['Druck [dezi mmHg]'] / 10.
            self._df['cPressure'] = self._df['Druck kompensiert [dezi mmHg]'] / 10.
            
            self._df.drop(['Druck [dezi mmHg]', 'Druck kompensiert [dezi mmHg]'], axis=1, inplace=True)
        else:
            self._df = df.copy()
        
        self._t_resolution = 0.004 if t_resolution is None else t_resolution
        
        self._points_df = pd.DataFrame()
        return
    
    @property
    def df(self):
        return self._df.copy()
    
    @property
    def points_df(self):
        return self._points_df.copy()
    
    def compute_derivatives(self, use_filter=True):
        self._sigma_filter_pressure = 6. # Orig (10.), Other: 
        self._df['fPressure'] = gaussian_filter1d(input=self.df['Pressure'].values, 
                                            sigma=self._sigma_filter_pressure)
        
        self._df['fcPressure']= gaussian_filter1d(input=self.df['cPressure'].values, 
                                            sigma=self._sigma_filter_pressure)
        
        self._sigma_filter_dpdt = 4
        self._df['dpdt'] = (
                            gaussian_filter1d(self._df['fcPressure'].values - np.roll(self._df['fcPressure'].values, shift=1), sigma=self._sigma_filter_dpdt) / self._t_resolution 
                            if use_filter else
                            ((np.roll(self._df['Pressure'].values, shift=-1) - np.roll(self._df['Pressure'].values, shift=1))/ self._t_resolution / 2.0)
                            )
        
        self._sigma_filter_d2pdt2 = 2 # Orig (2), Other: 1 
        self._df['d2pdt2'] = (
                            gaussian_filter1d(
                                            (np.roll(self._df['fcPressure'].values, shift=-1) - 2.0 * self._df['fcPressure'].values + np.roll(self._df['fcPressure'].values, shift=1)) / self._t_resolution / self._t_resolution,
                                            sigma = self._sigma_filter_d2pdt2
                                            )
                              if use_filter else
                                (np.roll(self._df['Pressure'].values, shift=-1) - 2.0 * self._df['Pressure'].values + np.roll(self._df['Pressure'].values, shift=1)) / self._t_resolution / self._t_resolution
                              )
        
        return
        
    def report_error_percentate(self):
        print(f"Percentage error: {self._df['Error'].sum() / len(self._df) * 100.:.2f}%")
        return
    
    def compute_points_of_interest(self, height=100, use_filter=True):
        # Compute anti-epad: the minimum dpdt 
        a_epad_ind, _ = find_peaks(gaussian_filter1d(-self._df['dpdt'].values, sigma=4), 
                                   height=height, 
                                   distance=100)
        self._points_df['a_epad_ind'] = a_epad_ind.astype(int)
        
        if not use_filter: 
            pressure = self._df['Pressure'].values.copy()
        else:
            pressure = self._df['fcPressure'].values.copy()
        
        epad_ind = np.zeros(a_epad_ind.shape, dtype=int)
        dia_ind  = np.zeros(a_epad_ind.shape, dtype=int)
        sys_ind  = np.zeros(a_epad_ind.shape, dtype=int)
        esp_ind  = np.zeros(a_epad_ind.shape, dtype=int)
        
        for i, a_epad in enumerate(a_epad_ind[:-1]):
            # Compute epad
            # temp, _ = find_peaks(self._df['dpdt'][a_epad:a_epad_ind[i+1]], height=height)
            temp = np.argmax(self._df['dpdt'][a_epad:a_epad_ind[i+1]])
            try:
                epad_ind[i] = int(temp[0]) + a_epad
            except:
                epad_ind[i] = a_epad + temp
                            
            # Compute dia
            temp = np.where(
                            (self._df['dpdt'][a_epad:a_epad_ind[i+1]] >= 0.0) 
                            & 
                            (pressure[a_epad:a_epad_ind[i+1]] <= pressure[a_epad:a_epad_ind[i+1]].min() + 10.)
                            )
            dia_ind[i] = int(temp[0][0]) + a_epad
            
            # Compute sys
            temp = np.where(self._df['dpdt'][epad_ind[i]:] < 0.0)
            sys_ind[i] = int(temp[0][0]) + epad_ind[i]
            
            # Computed esp
            temp = np.argmin(self._df['d2pdt2'][sys_ind[i]:a_epad_ind[i+1]])
            try:
                esp_ind[i] = temp[0] + sys_ind[i]
            except:
                esp_ind[i] = temp    + sys_ind[i]
            
        self._points_df['epad_ind'] = epad_ind
        self._points_df['dia_ind']  = dia_ind
        self._points_df['sys_ind']  = sys_ind
        self._points_df['esp_ind']  = esp_ind
        
        shift = 1
        temp = self._points_df.copy()
        temp['a_epad_ind'] = np.roll(temp['a_epad_ind'].values, shift=-shift)
        temp.drop(len(temp) - 1, inplace=True)
        del self._points_df
        self._points_df = temp
        
        
        self._points_df['a_epad']  = pressure[self._points_df['a_epad_ind'].values.astype(int)]
        self._points_df['epad']    = pressure[self._points_df['epad_ind'].values.astype(int)]
        
        try:
            self._points_df['s_a_epad']= pressure[self._points_df['a_epad_ind'].values.astype(int) + 3]
        except:
            self._points_df['s_a_epad']= pressure[self._points_df['a_epad_ind'].values.astype(int) + 3 - len(self._points_df['a_epad_ind'])]
        self._points_df['s_epad']  = pressure[self._points_df['epad_ind'].values.astype(int) - 3]
        
        self._points_df['a_alpha'] = (self._points_df['s_a_epad'] - self._points_df['a_epad']) / 3.
        self._points_df['b_alpha'] = self._points_df['a_epad'] - self._points_df['a_alpha'] * self._points_df['a_epad_ind']
        
        self._points_df['a_beta']  = (self._points_df['epad'] - self._points_df['s_epad']) / 3.
        self._points_df['b_beta']  = self._points_df['a_epad'] - self._points_df['a_beta'] * self._points_df['epad_ind']
        
        self._points_df['cross_ind'] = - (self._points_df['b_alpha'] - self._points_df['b_beta']) / (self._points_df['a_alpha'] - self._points_df['a_beta'])
        self._points_df['cross_max']     = self._points_df['a_beta'] * self._points_df['cross_ind'] + self.points_df['b_beta']
        
        self._points_df['A_p']     = (self._points_df['epad'] + self._points_df['a_epad']) / 2.
        self._points_df['P_max']   = (self._points_df['cross_max'] - self._points_df['A_p']) * 2. / np.pi + self._points_df['A_p']
        ####################################
        self._points_df['esp']     = pressure[self._points_df['esp_ind'].values.astype(int)]
        self._points_df['sys']     = pressure[self._points_df['sys_ind'].values.astype(int)]
        self._points_df['EF']      = 1.0 - self._points_df['esp'] / self._points_df['P_max']
        ####################################
        self._points_df['dia']     = pressure[self._points_df['dia_ind'].values.astype(int)]
        self._points_df['min_dpdt']= self._df['dpdt'].iloc[self._points_df['a_epad_ind'].values.astype(int)].values
        self._points_df['tau']     = -(self._points_df['a_epad'] - self._points_df['dia']) / 2.0 / self._points_df['min_dpdt']
        self._points_df['Ees/Ea']  = self._points_df['P_max'] / self._points_df['esp'] - 1.0
        #####################################
        self._points_df['iT']      = (self._points_df['dia_ind'].values - np.roll(self._points_df['dia_ind'].values, shift=1)) * 0.004
        self._points_df.loc[0, 'iT'] = 0
        #####################################
        self._points_df['iHR']      = 60. / self._points_df['iT']
        self._points_df.loc[0, 'iHR'] = 0
        
        return
        
    
    def plot_pressures(self, start=0, finish=-1):
        finish = len(self._df) + finish if finish == -1 else finish
        
        a_epad_ind = self._points_df['a_epad_ind'].values.astype(int)
        a_epad_ind = a_epad_ind[(a_epad_ind >= start) & (a_epad_ind < finish)]
        
        epad_ind = self._points_df['epad_ind'].values.astype(int)
        epad_ind = epad_ind[(epad_ind >= start) & (epad_ind < finish)]
        
        dia_ind = self._points_df['dia_ind'].values.astype(int)
        dia_ind = dia_ind[(dia_ind >= start) & (dia_ind < finish)]
        
        sys_ind = self._points_df['sys_ind'].values.astype(int)
        sys_ind = sys_ind[(sys_ind >= start) & (sys_ind < finish)]
        
        esp_ind = self._points_df['esp_ind'].values.astype(int)
        esp_ind = esp_ind[(esp_ind >= start) & (esp_ind < finish)]
        
        _, ax = plt.subplots(figsize=(20,15), nrows=5)

        ax[0].grid(axis='x')
        ax[0].plot(self._df.index[start:finish], self._df['cPressure'].iloc[start:finish] , label='Compensated', linewidth=4)
        ax[0].plot(self._df.index[start:finish], self._df['fcPressure'].iloc[start:finish], label='c. filtered', linewidth=4, linestyle='-.')
        ax[0].legend()
        
        for a_epad, epad, dia, sys, esp in zip(a_epad_ind, epad_ind, dia_ind, sys_ind, esp_ind):
            ax[0].axvline(self._df.index[a_epad], color=mcolors.TABLEAU_COLORS['tab:olive'], linewidth=4, linestyle=':')
            ax[0].axvline(self._df.index[epad],   color=mcolors.TABLEAU_COLORS['tab:blue'],  linewidth=4, linestyle=':')
            ax[0].axvline(self._df.index[dia],    color=mcolors.TABLEAU_COLORS['tab:red'],   linewidth=4, linestyle=':')
            ax[0].axvline(self._df.index[sys],    color=mcolors.TABLEAU_COLORS['tab:purple'],linewidth=4, linestyle=':')
            ax[0].axvline(self._df.index[esp],    color='r',                                 linewidth=1, linestyle='-')
        
        self._df['Noise']  = self._df['Pressure'] - self._df['fPressure']
        self._df['cNoise'] = self._df['cPressure'] - self._df['fcPressure']

        ax[1].grid(axis='x')
        ax[1].plot(self._df.index[start:finish], self._df['Noise'].iloc[start:finish], label='Noise', linewidth=4, linestyle='-')
        ax[1].plot(self._df.index[start:finish], self._df['cNoise'].iloc[start:finish], label='cNoise', linewidth=4, linestyle='-')
        ax[1].legend()
        
        self._df['Compensation']  = self._df['cPressure']  - self._df['Pressure']
        self._df['fCompensation'] = self._df['fcPressure'] - self._df['fPressure']

        ax[2].grid(axis='x')
        ax[2].plot(self._df.index[start:finish], self._df['Compensation'].iloc[start:finish] , label='Compensation', linewidth=4, linestyle='-')
        ax[2].plot(self._df.index[start:finish], self._df['fCompensation'].iloc[start:finish],label='Filtered compensation', linewidth=4, linestyle='-')
        ax[2].legend()
        
        ax[3].grid(axis='x')
        ax[3].plot(self._df.index[start:finish], self._df['dpdt'].iloc[start:finish] , label='$\\frac{dp}{dt}$', linewidth=4, linestyle='-')
        ax[3].legend()
        
        for a_epad, epad, dia, sys in zip(a_epad_ind, epad_ind, dia_ind, sys_ind):
            ax[3].axvline(self._df.index[a_epad], color=mcolors.TABLEAU_COLORS['tab:olive'], linewidth=4, linestyle=':')
            ax[3].axvline(self._df.index[epad],   color=mcolors.TABLEAU_COLORS['tab:blue'],  linewidth=4, linestyle=':')
            ax[3].axvline(self._df.index[dia],    color=mcolors.TABLEAU_COLORS['tab:red'],   linewidth=4, linestyle=':')
            

        ax[4].grid(axis='x')
        ax[4].plot(self._df.index[start:finish], self._df['d2pdt2'].iloc[start:finish] , label='$\\frac{d^2p}{dt^2}$', linewidth=4, linestyle='-')
        ax[4].legend()
        
        for sys, a_epad, esp in zip(sys_ind, a_epad_ind, esp_ind):
            ax[4].axvline(self._df.index[sys],    color=mcolors.TABLEAU_COLORS['tab:purple'],   linewidth=4, linestyle=':')
            ax[4].axvline(self._df.index[a_epad], color=mcolors.TABLEAU_COLORS['tab:olive'],   linewidth=4, linestyle=':')
            ax[4].axvline(self._df.index[esp],    color='r',                                   linewidth=1, linestyle='-')
        plt.show()
        return
    
    
    def plot_single_pulse_metrics(self):
        fig, ax = plt.subplots(nrows=5, figsize=(20, 20))
        
        ax[0].plot(self._points_df['dia'],  label='dia')
        ax[0].plot(self._points_df['sys'],  label='sys')
        ax[0].plot(self._points_df['epad'], label='epad')
        ax[0].plot(self._points_df['esp'],  label='esp')
        ax[0].set_xlim([0, len(self._points_df)])
        ax[0].set_ylabel('Pressure [mmHg]')
        ax[0].set_xlabel('Heart beat index')
        
        ax[0].legend()
        
        ax[1].plot(self._points_df['EF'], label='EF')        
        ax[1].set_xlim([0, len(self._points_df)])
        ax[1].set_ylabel('Ejection fraction')
        ax[1].set_xlabel('Heart beat index')
        
        ax[1].legend()
        
        ax[2].plot(self._points_df['tau'], label='tau')
        ax[2].set_xlim([0, len(self._points_df)])
        ax[2].set_ylabel('ms')
        ax[2].set_xlabel('Heart beat index')
        ax[2].legend()
        
        ax[3].plot(self._points_df['iT'], label='iT')
        ax[3].set_xlim([0, len(self._points_df)])
        ax[3].set_ylabel('Pulse duration [s]')
        ax[3].set_xlabel('Heart beat index')
        
        ax3_2  = ax[3].twinx()
        ax3_2.plot(self._points_df['iHR'], '-.r', label='iHR')
        ax3_2.set_ylabel('HR [beats/min]', color='tab:red')
        
        ax[3].legend()
        
        self._df['Acc'] = (self._df['ACC x [centi g]']**2.0 + self._df['ACC y [centi g]']**2.0 + self._df['ACC z [centi g]']**2.0) ** 0.5 / 100.
        
        ax[4].plot(self._df.index, self._df['Acc'], label='iT')
        ax[4].set_xlim([self._df.index[0], self._df.index[-1]])
        ax[4].set_xlabel('Time')
        ax[4].set_ylabel('Acc [G]')
        
        fig.tight_layout()
        plt.show()
        return
    
    def resample_heart_beat(self):
        dia_array = self._points_df['dia_ind'].values
        pulses = np.zeros((len(dia_array)-1, 101))
        for i, dia_indx in enumerate(dia_array[:-1]):
            ind1, ind2 = dia_indx, dia_array[i+1]
            pulses[i,:] = np.interp(np.linspace(0, ind2-ind1, num=101), np.linspace(0, ind2-ind1, ind2-ind1), self._df['fcPressure'].iloc[ind1:ind2])
        return pulses
