import numpy as np
import wfdb
import neurokit2 as nk
import pandas as pd
import os
from scipy.io import loadmat
from wfdb import processing

class get_features:

    def __init__(self, r_peak=True, r_int=True, p_peak=True, p_int=True, p_onset=True, t_peak=True, t_int=True, q_peak=True, q_int=True, s_peak=True, s_int=True, qrs_dur=True, qt_dur=True, pr_dur=True, search_radius=25, smooth_window_size=7, voltage_res=1):
        """
        Initialize the feature extraction settings.
        
        Parameters:
        - r_peak, r_int, ...: Enable/disable specific features.
        - search_radius (int): Radius used for peak correction.
        - smooth_window_size (int): Window size used for smoothing during peak correction.
        """
        self.rpeak_int = r_int
        self.ppeak_int = p_int
        self.tpeak_int = t_int
        self.qpeak_int = q_int
        self.speak_int = s_int
        
        self.rpeak_amp = r_peak
        self.ppeak_amp = p_peak
        self.tpeak_amp = t_peak
        self.qpeak_amp = q_peak
        self.speak_amp = s_peak
        self.ponset_amp = p_onset
        
        self.qrs_duration = qrs_dur
        self.qt_duration = qt_dur
        self.pr_duration = pr_dur

        self.search_radius = search_radius
        self.smooth_window_size = smooth_window_size
        self.voltage_res = voltage_res

    def featurize_ecg(self, recording, sample_freq):

        def interval_calc_simple(first_peak, second_peak, sample_freq):
            try:
                mean_interval = round((second_peak-first_peak).mean(),5)
            except:
                mean_interval = float("NaN")
            try:
                std_interval = round((second_peak-first_peak).std(),5)
            except:
                std_interval = float("NaN")
            return mean_interval, std_interval


        
        feature_list = []
        feature_name = []
        try:
            temp_data = nk.ecg_process(recording,sample_freq)[0]
            r_peaks = np.where(temp_data['ECG_R_Peaks']==1)[0]
            p_peaks = np.where(temp_data['ECG_P_Peaks']==1)[0]
            q_peaks = np.where(temp_data['ECG_Q_Peaks']==1)[0]
            s_peaks = np.where(temp_data['ECG_S_Peaks']==1)[0]
            t_peaks = np.where(temp_data['ECG_T_Peaks']==1)[0]
            p_onset = np.where(temp_data['ECG_P_Onsets']==1)[0]
            t_offset = np.where(temp_data['ECG_T_Offsets']==1)[0]

            analysis = True
        except:
            analysis = False
            r_peaks = np.array([1,2])
            p_peaks = np.array([1,2])
            q_peaks = np.array([1,2])
            s_peaks = np.array([1,2])
            t_peaks = np.array([1,2])
            p_onset = np.array([]) 

        
        if self.rpeak_int == True:
            feature_name.append("mean_rr_interval")
            feature_name.append("sd_rr_interval")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append((np.diff(r_peaks)/sample_freq).mean())
                feature_list.append((np.diff(r_peaks)/sample_freq).std())
            

        if self.rpeak_amp == True:
            feature_name.append("mean_r_peak")
            feature_name.append("sd_r_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append(recording[r_peaks].mean())
                feature_list.append(recording[r_peaks].std())

        if self.ppeak_int == True:
            feature_name.append("mean_pp_interval")
            feature_name.append("sd_pp_interval") 
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append((np.diff(p_peaks)/sample_freq).mean())
                feature_list.append((np.diff(p_peaks)/sample_freq).std())
         
        if self.ppeak_amp == True:
            feature_name.append("mean_p_peak")
            feature_name.append("sd_p_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append(temp_data['ECG_Clean'][p_peaks].mean())
                feature_list.append(temp_data['ECG_Clean'][p_peaks].std())

        if self.tpeak_int == True:
            feature_name.append("mean_tt_interval")
            feature_name.append("sd_tt_interval")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append((np.diff(t_peaks)/sample_freq).mean())
                feature_list.append((np.diff(t_peaks)/sample_freq).std())
         
        if self.tpeak_amp == True:
            feature_name.append("mean_t_peak")
            feature_name.append("sd_t_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                 
                feature_list.append(temp_data['ECG_Clean'][t_peaks].mean())
                feature_list.append(temp_data['ECG_Clean'][t_peaks].std())

        if self.qpeak_int == True:
            feature_name.append("mean_qq_interval")
            feature_name.append("sd_qq_interval") 
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True: 
                feature_list.append((np.diff(q_peaks)/sample_freq).mean())
                feature_list.append((np.diff(q_peaks)/sample_freq).std())
         
        if self.qpeak_amp == True:
            feature_name.append("mean_q_peak")
            feature_name.append("sd_q_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                         
                feature_list.append(temp_data['ECG_Clean'][q_peaks].mean())
                feature_list.append(temp_data['ECG_Clean'][q_peaks].std())

        if self.speak_int == True:
            feature_name.append("mean_ss_interval")
            feature_name.append("sd_ss_interval")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                        
                feature_list.append((np.diff(s_peaks)/sample_freq).mean())
                feature_list.append((np.diff(s_peaks)/sample_freq).std())
                                    
        if self.speak_amp == True:
            feature_name.append("mean_s_peak")
            feature_name.append("sd_s_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                        
                feature_list.append(temp_data['ECG_Clean'][s_peaks].mean())
                feature_list.append(temp_data['ECG_Clean'][s_peaks].std())

        if self.qrs_duration == True:
            feature_name.append("qrs_mean")
            feature_name.append("qrs_std")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan")) 
            elif analysis == True:                         
                qrs_mean, qrs_std = interval_calc_simple(q_peaks,s_peaks,sample_freq)
                feature_list.append(qrs_mean)
                feature_list.append(qrs_std)
                                    
        if self.qt_duration == True:
            feature_name.append("qt_mean")
            feature_name.append("qt_std")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                qt_mean, qt_std = interval_calc_simple(q_peaks,t_peaks,sample_freq)
                feature_list.append(qt_mean)
                feature_list.append(qt_std)

        if self.pr_duration == True:
            feature_name.append("pr_mean")
            feature_name.append("pr_std")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                pr_mean, pr_std = interval_calc_simple(p_peaks,r_peaks,sample_freq)
                feature_list.append(pr_mean)
                feature_list.append(pr_std)


        if self.rpeak_int == True:
            rr_intervals = np.diff(r_peaks) / sample_freq
            rmssd = np.mean(np.square(60 / rr_intervals))
            feature_name.extend(["RMSSD"])
            feature_list.extend([rmssd])

            # heartrate_r
            heartrate_r = (60 / rr_intervals)

            # heartrate_std
            heartrate_std = heartrate_r.std()
            feature_name.extend(["R_HR_STD"])
            feature_list.extend([heartrate_std])

            # heartrate_median
            heartrate_median = np.median(heartrate_r)
            feature_name.extend(["R_HR_median"])
            feature_list.extend([heartrate_median])

            # heartrate_min
            heartrate_min = heartrate_r.min()
            feature_name.extend(["R_HR_min"])
            feature_list.extend([heartrate_min])

            # heartrate_max
            heartrate_max = heartrate_r.max()
            feature_name.extend(["R_HR_max"])
            feature_list.extend([heartrate_max])

            # mean_heartrate_r
            mean_heartrate_r = heartrate_r.mean()
            feature_name.extend(["mean_heartrate_r"])
            feature_list.extend([mean_heartrate_r])

        if self.ponset_amp == True:
            if len(p_onset) == 0:
                ECG_baseline = float("nan")
            else:
                p_onset_values = recording[p_onset.astype(int)]
                ECG_baseline = p_onset_values.mean() / self.voltage_res
            feature_name.extend(["ECG_baseline"])
            feature_list.extend([ECG_baseline])


        feature_list = np.asarray(feature_list)
        feature_name = np.asarray(feature_name)
            
        feature_list = np.asarray(feature_list)
        feature_name = np.asarray(feature_name)
         
        return feature_list,feature_name, [p_peaks,q_peaks,r_peaks,s_peaks,t_peaks]
    
    def corr_and_featurize_ecg(self, recording, sample_freq, r_peaks, s_peaks, q_peaks, p_peaks, t_peaks):
        
        def interval_calc_simple(first_peak, second_peak, sample_freq):
            try:
                mean_interval = round((second_peak-first_peak).mean(),5)
            except:
                mean_interval = float("NaN")
            try:
                std_interval = round((second_peak-first_peak).std(),5)
            except:
                std_interval = float("NaN")
            return mean_interval, std_interval

        
        feature_list = []
        feature_name = []


        if len(r_peaks) and len(q_peaks) and len(s_peaks) and len(p_peaks) and len(t_peaks) < 3:
            try:
                temp_data = nk.ecg_process(recording,sample_freq)[0]
                r_peaks = np.where(temp_data['ECG_R_Peaks']==1)[0]
                p_peaks = np.where(temp_data['ECG_P_Peaks']==1)[0]
                q_peaks = np.where(temp_data['ECG_Q_Peaks']==1)[0]
                s_peaks = np.where(temp_data['ECG_S_Peaks']==1)[0]
                t_peaks = np.where(temp_data['ECG_T_Peaks']==1)[0]
                p_onset = np.where(temp_data['ECG_P_Onsets']==1)[0]
                t_offset = np.where(temp_data['ECG_T_Offsets']==1)[0]
                clean_rec = temp_data['ECG_Clean']

                analysis = True
            except:
                analysis = False
                r_peaks = np.array([1,2])
                p_peaks = np.array([1,2])
                q_peaks = np.array([1,2])
                s_peaks = np.array([1,2])
                t_peaks = np.array([1,2])
                p_onset = np.array([])  
        
        else:
            analysis = True
            clean_rec = nk.ecg_clean(recording)
            try:
                r_peaks = processing.peaks.correct_peaks(clean_rec, r_peaks, search_radius=self.search_radius, 
                                                     smooth_window_size=self.smooth_window_size, peak_dir='compare')
            except:
                r_peaks = r_peaks
            
            try:
                q_peaks = processing.peaks.correct_peaks(clean_rec, q_peaks, search_radius=25, 
                                                      smooth_window_size=7, peak_dir='compare')
            except:
                q_peaks = q_peaks
            
            try:
                s_peaks = processing.peaks.correct_peaks(clean_rec, s_peaks, search_radius=25, 
                                                          smooth_window_size=7, peak_dir='compare')
            except:
                s_peaks = s_peaks

            try:
                t_peaks = processing.peaks.correct_peaks(clean_rec, t_peaks, search_radius=25, 
                                                              smooth_window_size=7, peak_dir='compare')
            except:
                t_peaks = t_peaks
            
            try:
                p_peaks = processing.peaks.correct_peaks(clean_rec, p_peaks, search_radius=25, 
                                                              smooth_window_size=7, peak_dir='compare')
            except:
                p_peaks = p_peaks
            

        
        if self.rpeak_int == True:
            feature_name.append("mean_rr_interval")
            feature_name.append("sd_rr_interval")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append((np.diff(r_peaks)/sample_freq).mean())
                feature_list.append((np.diff(r_peaks)/sample_freq).std())
            

        if self.rpeak_amp == True:
            feature_name.append("mean_r_peak")
            feature_name.append("sd_r_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append(recording[r_peaks].mean())
                feature_list.append(recording[r_peaks].std())

        if self.ppeak_int == True:
            feature_name.append("mean_pp_interval")
            feature_name.append("sd_pp_interval") 
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append((np.diff(p_peaks)/sample_freq).mean())
                feature_list.append((np.diff(p_peaks)/sample_freq).std())
         
        if self.ppeak_amp == True:
            feature_name.append("mean_p_peak")
            feature_name.append("sd_p_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append(clean_rec[p_peaks].mean())
                feature_list.append(clean_rec[p_peaks].std())

        if self.tpeak_int == True:
            feature_name.append("mean_tt_interval")
            feature_name.append("sd_tt_interval")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                feature_list.append((np.diff(t_peaks)/sample_freq).mean())
                feature_list.append((np.diff(t_peaks)/sample_freq).std())
         
        if self.tpeak_amp == True:
            feature_name.append("mean_t_peak")
            feature_name.append("sd_t_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                 
                feature_list.append(clean_rec[t_peaks].mean())
                feature_list.append(clean_rec[t_peaks].std())

        if self.qpeak_int == True:
            feature_name.append("mean_qq_interval")
            feature_name.append("sd_qq_interval") 
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True: 
                feature_list.append((np.diff(q_peaks)/sample_freq).mean())
                feature_list.append((np.diff(q_peaks)/sample_freq).std())
         
        if self.qpeak_amp == True:
            feature_name.append("mean_q_peak")
            feature_name.append("sd_q_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                         
                feature_list.append(clean_rec[q_peaks].mean())
                feature_list.append(clean_rec[q_peaks].std())

        if self.speak_int == True:
            feature_name.append("mean_ss_interval")
            feature_name.append("sd_ss_interval")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                        
                feature_list.append((np.diff(s_peaks)/sample_freq).mean())
                feature_list.append((np.diff(s_peaks)/sample_freq).std())
                                    
        if self.speak_amp == True:
            feature_name.append("mean_s_peak")
            feature_name.append("sd_s_peak")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:                        
                feature_list.append(clean_rec[s_peaks].mean())
                feature_list.append(clean_rec[s_peaks].std())

        if self.qrs_duration == True:
            feature_name.append("qrs_mean")
            feature_name.append("qrs_std")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan")) 
            elif analysis == True:                         
                qrs_mean, qrs_std = interval_calc_simple(q_peaks,s_peaks,sample_freq)
                feature_list.append(qrs_mean)
                feature_list.append(qrs_std)
                                    
        if self.qt_duration == True:
            feature_name.append("qt_mean")
            feature_name.append("qt_std")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                qt_mean, qt_std = interval_calc_simple(q_peaks,t_peaks,sample_freq)
                feature_list.append(qt_mean)
                feature_list.append(qt_std)

        if self.pr_duration == True:
            feature_name.append("pr_mean")
            feature_name.append("pr_std")
            if analysis == False:
                feature_list.append(float("nan"))
                feature_list.append(float("nan"))
            elif analysis == True:
                pr_mean, pr_std = interval_calc_simple(p_peaks,r_peaks,sample_freq)
                feature_list.append(pr_mean)
                feature_list.append(pr_std)
        

        if self.rpeak_int == True:
            rr_intervals = np.diff(r_peaks) / sample_freq
            rmssd = np.mean(np.square(60 / rr_intervals))
            feature_name.extend(["RMSSD"])
            feature_list.extend([rmssd])

            # heartrate_r
            heartrate_r = (60 / rr_intervals)

            # heartrate_std
            heartrate_std = heartrate_r.std()
            feature_name.extend(["R_HR_STD"])
            feature_list.extend([heartrate_std])

            # heartrate_median
            heartrate_median = np.median(heartrate_r)
            feature_name.extend(["R_HR_median"])
            feature_list.extend([heartrate_median])

            # heartrate_min
            heartrate_min = heartrate_r.min()
            feature_name.extend(["R_HR_min"])
            feature_list.extend([heartrate_min])

            # heartrate_max
            heartrate_max = heartrate_r.max()
            feature_name.extend(["R_HR_max"])
            feature_list.extend([heartrate_max])

            # mean_heartrate_r
            mean_heartrate_r = heartrate_r.mean()
            feature_name.extend(["mean_heartrate_r"])
            feature_list.extend([mean_heartrate_r])

        if self.ponset_amp == True:
            if len(p_onset) == 0:
                ECG_baseline = float("nan")
            else:
                p_onset_values = recording[p_onset.astype(int)]
                ECG_baseline = p_onset_values.mean() / self.voltage_res
            feature_name.extend(["ECG_baseline"])
            feature_list.extend([ECG_baseline])
            
        feature_list = np.asarray(feature_list)
        feature_name = np.asarray(feature_name)
         
        return feature_list,feature_name, [p_peaks,q_peaks,r_peaks,s_peaks,t_peaks]