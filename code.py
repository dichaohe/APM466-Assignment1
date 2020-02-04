import math
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

def interpolate(data):
            x = np.array([0,1,4,5,6,7,8,11,12,13])
            df = pd.DataFrame() 
            for i in range(0,10):
                        close_price = data.iloc[i,6:20]
                        y = close_price.dropna().values
                        for j in [2,3,9,10]:
                                    f = interp1d(x,y,kind = 'linear')
                                    yj = f(j)
                                    yj = round(float(yj),2)
                                    close_price[j] = yj
                        df = df.append(close_price)
            return df

def compute_year_fraction():
            year_fraction = []
            d1 = datetime.datetime(2020,1,16)
            d2 = datetime.datetime(2020,3,1)
            d3 = datetime.datetime(2020,9,1)
            d4 = datetime.datetime(2021,3,1)
            d5 = datetime.datetime(2021,9,1)
            d6 = datetime.datetime(2022,3,1)
            d7 = datetime.datetime(2022,9,1)
            d8 = datetime.datetime(2023,3,1)
            d9 = datetime.datetime(2023,9,1)
            d10 = datetime.datetime(2024,3,1)
            d11 = datetime.datetime(2024,9,1)
            d12 = datetime.datetime(2025,3,1)
            d = [d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12]
            for i in range(len(d)):
                        interval = d[i] - d1
                        fraction = interval.days/365
                        year_fraction.append(fraction)
            return year_fraction

def ytm(bonds):
            bond_yield = {}
            #ytm1 = {}
            yr_frac = compute_year_fraction()
            for i in range(0,10):
                        ytm = []
                        #ytm11 = []#full_ytm_list
                        pv = []
                        #yr = year_fraction(date[i],input_dict)#year_fraction_dict
                        j = 0
                        while True:
                                    if ytm == []:
                                                break
                                    bond = bonds[j]
                                    next_bond = bonds[j+1]
                                    ytm = cal_ytm(bond,i,yr)
                                    ytm.append(ytm)
                                    #ytm11.append(ytm)
                                    if cal_year_diff(bond.maturitydate,next_bond.maturitydate)>0.6:
                                                next_bond_ytm = cal_ytm(next_bond,i,yr)#
                                                ytm_mean = (next_bond_ytm + ytm)/2#
                                                ytm.append(ytm_mean)
                                                ytm.append(next_bond_ytm)
                                                #ytm11.append(next_bond_ytm)
                                                j = j+1
                                    j = j+1
                                    ytm = cal_ytm(bonds_list[-1],i,yr)
                                    ytm.append(ytm)
                                    #ytm11.append(ytm2)
                                    #print(ytm00)
                                    #ytm[i] = ytm00
                                    #ytm1[i] = ytm1
                                    
                        plt.plot(yr_frac,bonds.iloc[:,i+21],label = i)
                        plt.legend()
                        plt.ylim(0,0.05)
            plt.title('yield curve')
            plt.show()
            return ytm,bond_yield,yr_frac

def present_value(fv,coupon,periods,rate,yr,spot_rate_list,gap):
            pv = 0
            #periods = periods_dict[periods]
            if gap == False:
                        for i in range(periods):
                                    total_pv = total_pv + coupon * math.exp(-yr[i]*spot_rate_list[i]/100)
                        pv = pv + fv * math.exp(-yr[i+1]*rate)
            else:
                        for i in range(periods - 1):
                                    pv = pv + coupon * math.exp(-yr[i]*spot_rate_list[i]/100)
                        pv = pv + coupon * math.exp(-yr[i+1]*(spot_rate_list[-1]/100*rate)/2)
                        pv = pv + fv*math.exp(-yr[i+2]*rate)
            return pv

def spot_rate(bonds):
            rate = True
            yr_frac = compute_year_fraction()
            spot_rate = {}
            for i in range(10):
                        while True:
                                    if rate:
                                                break
                                    pv = cal_pv(bond,day)
                                    fv = bond.coupon_rate + 100
                                    coupon = bond.coupon_rate
                                    periods = bond.periods
                                    spot_rate_one = ytm_list[bond_index]/100
                                    condition = True
                                    while True:
                                                if pv<fv:
                                                            spot_rate_one = spot_rate_one - 1e-6
                                                else:
                                                            spot_rate_one = spot_rate_one + 1e-6
                                                
                                                if pv<fv:
                                                            condition = total_pv_one<pv
                                                else:
                                                            condition = total_pv_one>pv
                                                pv = present_value(fv,coupon,periods,spot_rate_one,yr,spot_rate_list,gap)
                        plt.plot(yr_frac,bonds.iloc[:,i+32],label = i)
                        plt.ylim(0,0.05)
                        plt.legend()
            plt.title('spot curve')
            plt.show()
            return spot_rate

def forward_rate(bonds):
            forward_rate = {}
            x = [2,3,4,5,6,7,8,9,10,11,12]
            for i in range(10):
                        while True:
                                    forward_rate_list = []
                                    if forward_rate == {}:
                                                break
                                                ytm_list = ytm0[i]
                                                if j == 0:
                                                            forward_rate_list.append(ytm_list[0])
                                                if j >= 1:
                                                            prev_bond = bonds[j-1]
                                                            bond = bonds[j]
                                                            forward_rate = cal_forward_rate(bond,i,j,yr_frac[i],forward_rate_list,ytm_list,True)
                                                            #forward_rate_list.append((forward_rate_list[-1]+spot_rate)/2)
                                                            forward_rate_list.append(forward_rate)
                                                forward_rate_dict[i] = forward_rate_list
                        plt.plot(x,bonds.iloc[:,i+43],label = i)
                        plt.ylim(0,0.05)
                        plt.legend()
            plt.title('forward curve')
            plt.show()
            return forward_rate

def covariance_matrix(bonds):
            log_return = np.array([])
            for i in range(5):
                        yld = bonds.iloc[:,i+21]
                        cal_log_return = np.log(yld.pct_change()+1).dropna().values
                        if log_return  is np.array([]):
                                    log_return  = cal_log_return
                        else:
                                    log_return  = np.append(log_return,cal_log_return)
            log_return  = log_return.reshape(5,10)
            cov_mat1 = np.cov(log_return)
            print(cov_mat1)
            forward_rate = bonds.iloc[0:4,31:41].values
            cov_mat2 = np.cov(forward_rate)
            print(cov_mat2)
            a,b = np.linalg.eig(cov_mat1)
            print('log return matrix:\n engivalue {}\n engivector\n{}'.format(a,b))
            c,d = np.linalg.eig(cov_mat2)
            print('forward rates matrix:\n engivalue {}\n engivector\n{}'.format(c,d))


if __name__ == '__main__':
            bonds = pd.read_csv('data.csv')
            #df = interpolate(bonds)
            #df.to_csv('close_price.csv')
            ytm(bonds)
            spot_rate(bonds)
            forward_rate(bonds)
            covariance_matrix(bonds)
            
                                    
                        
            
            
