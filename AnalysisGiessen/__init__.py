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
            self._df['Temperature'] = self.df['Sondentemperatur [dezi °C]'] / 10.
            
            self._df.drop(['Druck [dezi mmHg]', 'Druck kompensiert [dezi mmHg]'], axis=1, inplace=True)
            
        else:
            self._df = df.copy()
        
        if 'ACC x [centi g]' not in self._df.columns:
            self._df['ACC x [centi g]'] = 0.0
            self._df['ACC y [centi g]'] = 0.0
            self._df['ACC z [centi g]'] = 0.0
        if 'Temperature' not in self._df.columns:
            self._df['Temperature'] = 0.0
        
        self._t_resolution = 0.004 if t_resolution is None else t_resolution
        
        self._points_df = pd.DataFrame()
        self.epad_buffer = 10
        
        self._sigma_filter_pressure = 6.
        self._sigma_filter_dpdt = 4
        self._sigma_filter_d2pdt2 = 2
        return
    
    @property
    def df(self):
        return self._df.copy()
    
    @property
    def points_df(self):
        return self._points_df.copy()
    
    @property
    def sigma_filter_pressure(self):
        return self._sigma_filter_pressure
    
    @sigma_filter_pressure.setter
    def sigma_filter_pressure(self, value):
        self._sigma_filter_pressure = value
        
    @property
    def sigma_filter_dpdt(self):
        return self._sigma_filter_dpdt
    
    @sigma_filter_dpdt.setter
    def sigma_filter_dpdt(self, value):
        self._sigma_filter_dpdt = value
        
    @property
    def sigma_filter_d2pdt2(self):
        return self._sigma_filter_d2pdt2
    
    @sigma_filter_d2pdt2.setter
    def sigma_filter_d2pdt2(self, value):
        self._sigma_filter_d2pdt2 = value
    
    def compute_derivatives(self):
        self._df['fPressure'] = gaussian_filter1d(input=self.df['Pressure'].values, 
                                            sigma=self._sigma_filter_pressure)
        
        self._df['fcPressure']= gaussian_filter1d(input=self.df['cPressure'].values, 
                                            sigma=self._sigma_filter_pressure)
        
        self._df['dpdt']  = (np.roll(self._df['Pressure'].values, shift=-1) - np.roll(self._df['Pressure'].values, shift=1))/ self._t_resolution / 2.0
        self._df['fdpdt'] = gaussian_filter1d(np.roll(self._df['fcPressure'].values, shift=-1) - np.roll(self._df['fcPressure'].values, shift=1), 
                                              sigma=self._sigma_filter_dpdt) / self._t_resolution / 2.0
        
        # Orig (2), Other: 1 
        self._df['d2pdt2']  = (np.roll(self._df['Pressure'].values, shift=-1) - 2.0 * self._df['Pressure'].values + np.roll(self._df['Pressure'].values, shift=1)) / self._t_resolution / self._t_resolution
        self._df['fd2pdt2'] = gaussian_filter1d(
                                            (np.roll(self._df['fcPressure'].values, shift=-1) - 2.0 * self._df['fcPressure'].values + np.roll(self._df['fcPressure'].values, shift=1)) / self._t_resolution / self._t_resolution,
                                            sigma = self._sigma_filter_d2pdt2
                                            )
        
        return
        
    def report_error_percentate(self):
        print(f"Percentage error: {self._df['Error'].sum() / len(self._df) * 100.:.2f}%")
        return
    
    def compute_points_of_interest(self, height=100, height_d2pdt2=1000, distance=5, use_filter=True, export_true_derivates=False, exclusion_list=['dia'], export_true_p=False, start_at_edp=False):
        # Compute anti-epad: the minimum dpdt 
        a_epad_ind, _ = find_peaks(-self._df['fdpdt'], height=height, distance=100)
        self._points_df['a_epad_ind'] = a_epad_ind.astype(np.int64)
        
        self.start_at_edp = start_at_edp
        
        if not use_filter: 
            pressure_ind = self._df['cPressure'].values.copy()
            dpdt_4_ind   = self._df['dpdt'].values.copy()
            d2pdt2_4_ind = self._df['d2pdt2'].values.copy()
        else:
            pressure_ind = self._df['fcPressure'].values.copy()
            dpdt_4_ind   = self._df['fdpdt'].values.copy()
            d2pdt2_4_ind = self._df['fd2pdt2'].values.copy()
            
        if export_true_derivates:
            dpdt_4_exp   = self._df['dpdt'].values.copy()
        else:
            dpdt_4_exp   = self._df['fdpdt'].values.copy()
            
        if export_true_p:
            pressure_exp = self._df['cPressure'].values.copy()
        else:
            pressure_exp = self._df['fcPressure'].values.copy()
        
        epad_ind = np.zeros(a_epad_ind.shape, dtype=np.int64)
        dia_ind  = np.zeros(a_epad_ind.shape, dtype=np.int64)
        sys_ind  = np.zeros(a_epad_ind.shape, dtype=np.int64)
        esp_ind  = np.zeros(a_epad_ind.shape, dtype=np.int64)
        edp_ind  = np.zeros(a_epad_ind.shape, dtype=np.int64)
        eivc_ind = np.zeros(a_epad_ind.shape, dtype=np.int64)
        
        for i, a_epad in enumerate(a_epad_ind[:-1]):
            # Compute epad
            temp = np.argmax(dpdt_4_ind[(a_epad+self.epad_buffer):a_epad_ind[i+1]])
            try:
                epad_ind[i] = int(temp[0]) + a_epad + self.epad_buffer
            except:
                epad_ind[i] = a_epad + temp + self.epad_buffer
                            
            # Compute dia
            if 'dia' not in exclusion_list:
                temp = np.where(
                            (dpdt_4_ind[a_epad:a_epad_ind[i+1]] >= 0.0) 
                            & 
                            (pressure_ind[a_epad:a_epad_ind[i+1]] <= pressure_ind[a_epad:a_epad_ind[i+1]].min() + 10.)
                            )
            else:
                temp_dpdt = self._df['dpdt'].values.copy()
                temp = np.where(
                            (temp_dpdt[a_epad:a_epad_ind[i+1]] >= 0.0) 
                            & 
                            (pressure_ind[a_epad:a_epad_ind[i+1]] <= pressure_ind[a_epad:a_epad_ind[i+1]].min() + 10.)
                            )
            try:
                dia_ind[i] = int(temp[0][0]) + a_epad
            except:
                dia_ind[i] = a_epad
            
            # Compute sys
            temp = np.argmax(pressure_ind[epad_ind[i]:a_epad_ind[i+1]])
            try:
                sys_ind[i] = temp[0] + epad_ind[i]
            except:
                sys_ind[i] = temp    + epad_ind[i]
            
            # Compute esp
            if 'esp' not in exclusion_list:
                field = -d2pdt2_4_ind[sys_ind[i]:a_epad_ind[i+1]]
            else:
                field = -self._df['d2pdt2'].values[sys_ind[i]:a_epad_ind[i+1]]
            height = np.max(field) / 2.
            height = height if height > height_d2pdt2 else height_d2pdt2
            temp = []
            k = 0 
            while len(temp) == 0 and k < 10:
                temp, _ = find_peaks(field, height=height)
                height = height * 0.8
                k += 1
                if isinstance(temp, np.int64) : temp = [temp,]
            try:
                temp2   = np.argmin(pressure_ind[sys_ind[i] + temp])
                esp_ind[i] = temp[temp2] + sys_ind[i]
            except:
                temp, _ = find_peaks(-self._df['d2pdt2'].values[sys_ind[i]:a_epad_ind[i+1]], height=height_d2pdt2)
                try:
                    temp2   = np.argmin(pressure_ind[sys_ind[i] + temp])
                    esp_ind[i] = temp[temp2] + sys_ind[i]
                except:
                    esp_ind[i] = sys_ind[i]
            
            # Compute edp
            if 'edp' not in exclusion_list:
                temp, _ = find_peaks(d2pdt2_4_ind[dia_ind[i]:epad_ind[i]], height=height_d2pdt2, distance=distance)
            else:
                temp, _ = find_peaks(self._df['d2pdt2'].values[dia_ind[i]:epad_ind[i]], height=height_d2pdt2, distance=distance)
            try:
                temp2   = np.argmax(pressure_ind[dia_ind[i] + temp])
                if isinstance(temp2, np.int64):
                    edp_ind[i] = temp[temp2] + dia_ind[i]
                else:
                    edp_ind[i] = temp[temp2[0]] + dia_ind[i]
            except:
                if not isinstance(temp, np.ndarray):
                    edp_ind[i] = temp    + dia_ind[i]
                else:
                    edp_ind[i] = dia_ind[i]
                    
            # Compute eivc
            temp  = []
            field = -d2pdt2_4_ind[epad_ind[i]:sys_ind[i]]
            while len(temp) == 0 and k < 10:
                temp, _ = find_peaks(field, height=height)
                height = height * 0.8
                k += 1
                if isinstance(temp, np.int64) : temp = [temp,]
            # raise Exception(epad_ind[i], sys_ind[i], temp)
            try:
                temp2   = np.argmin(pressure_ind[epad_ind[i] + temp])
                eivc_ind[i] = temp[temp2] + epad_ind[i]
            except:
                temp, _ = find_peaks(-self._df['d2pdt2'].values[epad_ind[i]:sys_ind[i+1]], height=height_d2pdt2)
                try:
                    temp2   = np.argmin(pressure_ind[epad_ind[i] + temp])
                    eivc_ind[i] = temp[temp2] + epad_ind[i]
                except:
                    eivc_ind[i] = epad_ind[i]
                
        self._points_df['epad_ind'] = epad_ind
        self._points_df['dia_ind']  = dia_ind
        self._points_df['sys_ind']  = sys_ind
        self._points_df['esp_ind']  = esp_ind
        self._points_df['edp_ind']  = edp_ind
        self._points_df['eivc_ind'] = eivc_ind
        
        shift = 1
        temp = self._points_df.copy()
        temp['a_epad_ind'] = np.roll(temp['a_epad_ind'].values, shift=-shift)
        
        if self.start_at_edp:  temp['dia_ind'] = np.roll(temp['dia_ind'].values, shift=-shift)
        temp.drop(len(temp) - 1, inplace=True)
        del self._points_df
        self._points_df = temp
        ref = self.points_df['dia_ind']
        if self.start_at_edp: ref = self.points_df['edp_ind']
        
        self._points_df['t_max_dpdt'] = (self._points_df['epad_ind'] - ref) * self._t_resolution
        self._points_df['t_min_dpdt'] = (self._points_df['a_epad_ind'] - ref) * self._t_resolution
        self._points_df['t_max_p']    = (self._points_df['sys_ind'] - ref) * self._t_resolution
        self._points_df['t_dia']      = (self._points_df['dia_ind'] - ref) * self._t_resolution
        
        self._points_df['a_epad']  = pressure_exp[self._points_df['a_epad_ind'].values.astype(int)]
        self._points_df['epad']    = pressure_exp[self._points_df['epad_ind'].values.astype(int)]
        
        try:
            self._points_df['s_a_epad']= pressure_exp[self._points_df['a_epad_ind'].values.astype(int) + 3]
        except:
            self._points_df['s_a_epad']= pressure_exp[self._points_df['a_epad_ind'].values.astype(int) + 3 - len(pressure_exp)]
        self._points_df['s_epad']  = pressure_exp[self._points_df['epad_ind'].values.astype(int) - 3]
        
        ################################
        self._points_df['min_dpdt']= dpdt_4_exp[self._points_df['a_epad_ind'].values.astype(int)]
        self._points_df['max_dpdt']= dpdt_4_exp[self._points_df['epad_ind'].values.astype(int)]
        ################################
        self._points_df['a_alpha'] = self._points_df['min_dpdt'] * self._t_resolution
        self._points_df['b_alpha'] = self._points_df['a_epad'] - self._points_df['a_alpha'] * self._points_df['a_epad_ind']
        ################################
        self._points_df['a_beta'] = self._points_df['max_dpdt'] * self._t_resolution
        self._points_df['b_beta'] = self._points_df['epad'] - self._points_df['a_beta'] * self._points_df['epad_ind']
        ################################
        self._points_df['cross_ind'] = - (self._points_df['b_alpha'] - self._points_df['b_beta']) / (self._points_df['a_alpha'] - self._points_df['a_beta'])
        self._points_df['cross_max']     = self._points_df['a_beta'] * self._points_df['cross_ind'] + self.points_df['b_beta']
        
        self._points_df['A_p']     = (self._points_df['epad'] + self._points_df['a_epad']) / 2.
        self._points_df['P_max']   = (self._points_df['cross_max'] - self._points_df['A_p']) * 2. / np.pi + self._points_df['A_p']
        ####################################
        self._points_df['esp']     = pressure_exp[self._points_df['esp_ind'].values.astype(int)]
        self._points_df['sys']     = pressure_exp[self._points_df['sys_ind'].values.astype(int)]
        self._points_df['EF']      = 1.0 - self._points_df['esp'] / self._points_df['P_max']
        ####################################
        self._points_df['dia']     = pressure_exp[self._points_df['dia_ind'].values.astype(int)]
        self._points_df['tau']     = -(self._points_df['a_epad'] - self._points_df['dia']) / 2.0 / self._points_df['min_dpdt']
        self._points_df['Ees/Ea']  = self._points_df['P_max'] / self._points_df['esp'] - 1.0
        #####################################
        self._points_df['iT']      = (np.roll(ref.values, shift=-1) - ref.values) * self._t_resolution
        self._points_df.loc[len(self._points_df)-1, 'iT'] = (len(pressure_exp) - 1 - ref.values[-1])* self._t_resolution
        #####################################
        self._points_df['iHR']     = 60. / self._points_df['iT']
        #####################################
        self._points_df['edp']     = pressure_exp[self._points_df['edp_ind'].values.astype(int)] 
        #####################################
        self._points_df['eivc']    = pressure_exp[self._points_df['eivc_ind'].values.astype(int)] 
        #####################################
        
        return
    
    def compute_points_of_interest_2(self, height=40, height_dpdt=100, height_d2pdt2=1000, distance=90, sim_len=100, mask=None):
        pressure4sys = self._df['fcPressure'].copy()
        if mask is not None: pressure4sys[mask] = pressure4sys[0]
        temp, temp2 = find_peaks(pressure4sys, distance=distance, height=height)
        self._points_df['sys_ind'] = temp.astype(np.int64) 
        self._points_df['sys']     = temp2['peak_heights'].astype(np.float64)
        
        self._points_df['edp_ind'] = np.arange(0, len(self._df['fcPressure']), sim_len)
        self._points_df['edp']     = self._df['fcPressure'][self._points_df['edp_ind']].values
        
        temp, temp2 = find_peaks(-self._df['fdpdt'], height=height_dpdt, distance=distance)
        self._points_df['a_epad_ind'] = temp.astype(np.int64)
        self._points_df['a_epad']     = self._df['fcPressure'][temp].values
        self._points_df['min_dpdt']   = -temp2['peak_heights'].astype(np.float64)
        
        
        self._points_df['epad_ind'] = 0
        self._points_df['dia_ind']  = 0
        self._points_df['eivc_ind'] = 0
        self._points_df['esp_ind']  = 0
        self._points_df['max_dpdt'] = 0
        
        pfield   = self._df['fcPressure'].values.copy()
        dpfield  = self._df['fdpdt'].values.copy()
        d2pfield = self._df['fd2pdt2'].values.copy()
        
        for i in range(len(self._points_df)):
            edp_ind = self._points_df.loc[i, 'edp_ind']
            sys_ind = self._points_df.loc[i, 'sys_ind']
            temp, temp2 = find_peaks(self._df['fdpdt'].values[edp_ind:sys_ind], height=height_dpdt, distance=distance)
            try:
                self._points_df['epad_ind'].values[i] = int(temp[0]) + edp_ind
                self._points_df['max_dpdt'].values[i] = temp2['peak_heights'][0]
            except:
                self._points_df['epad_ind'].values[i] = sys_ind
                self._points_df['max_dpdt'].values[i] = 0.0
            
            a_epad_ind_i   = self._points_df.loc[i, 'a_epad_ind']
            try:
                a_epad_ind_i_1 = self._points_df.loc[i+1, 'a_epad_ind']
            except:
                a_epad_ind_i_1 = -1 
            temp = np.where((dpfield[a_epad_ind_i:(i+1)*sim_len] >= 0.0) & (pfield[a_epad_ind_i:a_epad_ind_i_1] <= pfield[a_epad_ind_i:a_epad_ind_i_1].min() + 10.))
            try:
                self._points_df['dia_ind'].values[i] = int(temp[0][0]) + a_epad_ind_i
            except:
                self._points_df['dia_ind'].values[i] = a_epad_ind_i
                
            epad_ind = self._points_df.loc[i, 'epad_ind']
            temp, temp2 = find_peaks(-d2pfield[epad_ind:sys_ind], height=height_d2pdt2)
            try:
                self._points_df['eivc_ind'].values[i] = int(temp[0]) + epad_ind
            except:
                self._points_df['eivc_ind'].values[i] = epad_ind
                
            a_epad_ind = self._points_df.loc[i,'a_epad_ind']
            temp, temp2 = find_peaks(-d2pfield[sys_ind:a_epad_ind], height=height_d2pdt2)
            try:
                self._points_df['esp_ind'].values[i] = int(temp[0]) + sys_ind
            except:
                self._points_df['esp_ind'].values[i] = sys_ind
            
        self._points_df['dia']   = pfield[self._points_df['dia_ind'].values]
        self._points_df['esp']   = pfield[self._points_df['esp_ind'].values]
        self._points_df['eivc']  = pfield[self._points_df['eivc_ind'].values]
        self._points_df['epad']  = pfield[self._points_df['epad_ind'].values]
        
        ################################
        self._points_df['a_alpha'] = self._points_df['min_dpdt'] * self._t_resolution
        self._points_df['b_alpha'] = self._points_df['a_epad'] - self._points_df['a_alpha'] * self._points_df['a_epad_ind']
        ################################
        self._points_df['a_beta'] = self._points_df['max_dpdt'] * self._t_resolution
        self._points_df['b_beta'] = self._points_df['epad'] - self._points_df['a_beta'] * self._points_df['epad_ind']
        ################################
        self._points_df['cross_ind'] = - (self._points_df['b_alpha'] - self._points_df['b_beta']) / (self._points_df['a_alpha'] - self._points_df['a_beta'])
        self._points_df['cross_max']     = self._points_df['a_beta'] * self._points_df['cross_ind'] + self.points_df['b_beta']
        
        self._points_df['A_p']     = (self._points_df['epad'] + self._points_df['a_epad']) / 2.
        self._points_df['P_max']   = (self._points_df['cross_max'] - self._points_df['A_p']) * 2. / np.pi + self._points_df['A_p']
        ####################################
        self._points_df['EF']      = 1.0 - self._points_df['esp'] / self._points_df['P_max']
        
        ref = self._points_df['edp_ind']
        self._points_df['t_max_dpdt'] = (self._points_df['epad_ind'] - ref) * self._t_resolution
        self._points_df['t_min_dpdt'] = (self._points_df['a_epad_ind'] - ref) * self._t_resolution
        self._points_df['t_max_p']    = (self._points_df['sys_ind'] - ref) * self._t_resolution
        self._points_df['t_dia']      = (self._points_df['dia_ind'] - ref) * self._t_resolution
        
        self._points_df['t_sys'] = (self._points_df['sys_ind'] - ref) * self._t_resolution
        self._points_df['t_esp'] = (self._points_df['esp_ind'] - ref) * self._t_resolution
        self._points_df['t_eivc']= (self._points_df['eivc_ind']- ref) * self._t_resolution
        
    def plot_pressures(self, start=0, finish=-1, non_filter=True, plot_features=True, fontsize=10):
        finish = len(self._df) + finish if finish <= -1 else finish
        
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
        
        edp_ind = self._points_df['edp_ind'].values.astype(int)
        edp_ind = edp_ind[(edp_ind >= start) & (edp_ind < finish)]
        
        eivc_ind = self._points_df['eivc_ind'].values.astype(int)
        eivc_ind = eivc_ind[(eivc_ind >= start) & (eivc_ind < finish)]
        
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(20,21), nrows=7)

        ax[0].grid(axis='x')
        ax[0].plot(self._df.index[start:finish], self._df['cPressure'].iloc[start:finish] , label='Compensated', linewidth=4)
        ax[0].plot(self._df.index[start:finish], self._df['fcPressure'].iloc[start:finish], label='c. filtered', linewidth=4, linestyle='-.')
        ax[0].set_ylabel('Pressure [mmHg]')
        ax[0].legend()
        
        if plot_features:
            for a_epad, epad, dia, sys, esp, edp, eivc in zip(a_epad_ind, epad_ind, dia_ind, sys_ind, esp_ind, edp_ind, eivc_ind):
                ax[0].axvline(self._df.index[a_epad], color=mcolors.TABLEAU_COLORS['tab:olive'], linewidth=4, linestyle=':')
                ax[0].axvline(self._df.index[epad],   color=mcolors.TABLEAU_COLORS['tab:blue'],  linewidth=4, linestyle=':')
                ax[0].axvline(self._df.index[dia],    color=mcolors.TABLEAU_COLORS['tab:red'],   linewidth=4, linestyle=':')
                ax[0].axvline(self._df.index[sys],    color=mcolors.TABLEAU_COLORS['tab:purple'],linewidth=4, linestyle=':')
                ax[0].axvline(self._df.index[esp],    color='r',                                 linewidth=1, linestyle='-')
                ax[0].axvline(self._df.index[edp],    color='g',                                 linewidth=1, linestyle='-')
                ax[0].axvline(self._df.index[eivc],   color='m',                                 linewidth=1, linestyle='-')
        
        self._df['Noise']  = self._df['Pressure'] - self._df['fPressure']
        self._df['cNoise'] = self._df['cPressure'] - self._df['fcPressure']

        ax[1].grid(axis='x')
        ax[1].plot(self._df.index[start:finish], self._df['Noise'].iloc[start:finish], label='Noise', linewidth=4, linestyle='-')
        ax[1].plot(self._df.index[start:finish], self._df['cNoise'].iloc[start:finish], label='cNoise', linewidth=4, linestyle='-')
        ax[1].set_ylabel('Pressure [mmHg]')
        ax[1].legend()
        
        self._df['Compensation']  = self._df['cPressure']  - self._df['Pressure']
        self._df['fCompensation'] = self._df['fcPressure'] - self._df['fPressure']

        ax[2].grid(axis='x')
        ax[2].plot(self._df.index[start:finish], self._df['Compensation'].iloc[start:finish] , label='Compensation', linewidth=4, linestyle='-')
        ax[2].plot(self._df.index[start:finish], self._df['fCompensation'].iloc[start:finish],label='Filtered compensation', linewidth=4, linestyle='-')
        ax[2].legend()
        
        ax[3].grid(axis='x')
        ax[3].plot(self._df.index[start:finish], self._df['fdpdt'].iloc[start:finish] , label='$\\frac{dp}{dt}$', linewidth=4, linestyle='-')
        if non_filter :  ax[3].plot(self._df.index[start:finish], self._df['dpdt'].iloc[start:finish] , label='$\\frac{dp}{dt}$', linewidth=4, linestyle='--')
        ax[3].set_ylabel('$mmHg/s$')
        ax[3].legend()

        if plot_features:
            for a_epad, epad, dia, sys in zip(a_epad_ind, epad_ind, dia_ind, sys_ind):
                ax[3].axvline(self._df.index[a_epad], color=mcolors.TABLEAU_COLORS['tab:olive'], linewidth=4, linestyle=':')
                ax[3].axvline(self._df.index[epad],   color=mcolors.TABLEAU_COLORS['tab:blue'],  linewidth=4, linestyle=':')
                ax[3].axvline(self._df.index[dia],    color=mcolors.TABLEAU_COLORS['tab:red'],   linewidth=4, linestyle=':')
                ax[3].axvline(self._df.index[sys],    color=mcolors.TABLEAU_COLORS['tab:purple'],  linewidth=4, linestyle=':')


        ax[4].grid(axis='x')
        ax[4].plot(self._df.index[start:finish], self._df['fd2pdt2'].iloc[start:finish] , label='$\\frac{d^2p}{dt^2}$', linewidth=4, linestyle='-')
        if non_filter : ax[4].plot(self._df.index[start:finish], self._df['d2pdt2'].iloc[start:finish] , label='$\\frac{d^2p}{dt^2}$', linewidth=4, linestyle='--')
        ax[4].set_ylabel('$mmHg/s^2$')
        ax[4].legend()
        
        if plot_features:
            for sys, a_epad, esp, edp, eivc in zip(sys_ind, a_epad_ind, esp_ind, edp_ind, eivc_ind):
                # ax[4].axvline(self._df.index[a_epad], color=mcolors.TABLEAU_COLORS['tab:olive'],   linewidth=4, linestyle=':')
                ax[4].axvline(self._df.index[esp],    color='r',                                   linewidth=1, linestyle='-')
                ax[4].axvline(self._df.index[edp],    color='g',                                   linewidth=1, linestyle='-')
                ax[4].axvline(self._df.index[eivc],   color='m',                                   linewidth=1, linestyle='-')
                
        self._df['Acc'] = (self._df['ACC x [centi g]']**2.0 + self._df['ACC y [centi g]']**2.0 + self._df['ACC z [centi g]']**2.0) ** 0.5 / 100.
        
        ax[5].plot(self._df.index[start:finish], self._df['Acc'].iloc[start:finish], label='Acceleration')
        ax[5].grid(axis='x')
        ax[5].set_ylabel('Acc [G]')
        
        ax[6].plot(self._df.index[start:finish], self._df['Temperature'].iloc[start:finish], label='Temp.')
        ax[6].set_ylabel('Temp. [$^oC$]')
        ax[6].set_xlabel('Time [H:M:S]')
        ax[6].grid(axis='x')

        fig.tight_layout()
        plt.show()
        return
    
    
    
    def plot_single_pulse_metrics(self, start=0, finish=-1, fontsize=10):
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(nrows=4, figsize=(20, 12))
        finish = len(self._df) + finish if finish <= -1 else finish
        
        flag = self._points_df.query('dia_ind >= @start & dia_ind <= @finish').index
        # raise Exception(flag)
        # raise Exception((self._points_df['dia_ind'] >= start) & (self._points_df['dia_ind'] <= finish))
        ax[0].plot(self._points_df.loc[flag, 'dia'],  label='dia')
        ax[0].plot(self._points_df.loc[flag, 'sys'],  label='sys')
        ax[0].plot(self._points_df.loc[flag, 'epad'], label='epad')
        ax[0].plot(self._points_df.loc[flag, 'esp'],  label='esp')
        ax[0].plot(self._points_df.loc[flag, 'edp'],  label='edp')
        ax[0].plot(self._points_df.loc[flag, 'eivc'], label='eivc')
        ax[0].set_xlim([0, len(flag)-1])
        ax[0].set_ylabel('Pressure [mmHg]')
        ax[0].set_xlabel('Heart beat index')
        ax[0].grid(axis='x')
        ax[0].legend()
        
        ax[1].plot(self._points_df.loc[flag,'EF'], label='EF')        
        ax[1].set_xlim([0, len(flag)-1])
        ax[1].set_ylabel('Ejection fraction')
        ax[1].set_xlabel('Heart beat index')
        ax[1].grid(axis='x')
        ax[1].legend()
        
        ax[2].plot(self._points_df.loc[flag,'tau'], label='tau')
        ax[2].set_xlim([0, len(flag)-1])
        ax[2].set_ylabel('ms')
        ax[2].set_xlabel('Heart beat index')
        ax[2].grid(axis='x')
        ax[2].legend()
        
        ax[3].plot(self._points_df.loc[flag,'iT'], label='iT')
        ax[3].set_xlim([0, len(flag)-1])
        ax[3].set_ylabel('Pulse duration [s]')
        ax[3].set_xlabel('Heart beat index')
        ax[3].grid(axis='x')
        ax3_2  = ax[3].twinx()
        ax3_2.plot(self._points_df['iHR'], '-.r', label='iHR')
        ax3_2.set_ylabel('HR [beats/min]', color='tab:red')
        
        ax[3].legend()
        
        fig.tight_layout()
        plt.show()
        return
    
    def resample_heart_beat(self):
        if self.start_at_edp:
            ind_array = self._points_df['edp_ind'].values
        else:
            ind_array = self._points_df['dia_ind'].values
        pulses = np.zeros((len(ind_array)-1, 101))
        for i, indx in enumerate(ind_array[:-1]):
            ind1, ind2 = indx, ind_array[i+1]
            pulses[i,:] = np.interp(np.linspace(0, ind2-ind1, num=101), np.linspace(0, ind2-ind1, ind2-ind1), self._df['fcPressure'].iloc[ind1:ind2])
        return pulses
