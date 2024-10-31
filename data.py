import pandas as pd
import utils
from setting import *
from sklearn.preprocessing import StandardScaler

def create_unit_profile(csv_all):
    # Process the solar and wind data from solar_wind_data.csv

    pv_df = csv_all[['mu_solar(kW/m^2)','sigma_solar(kW/m^2)']]
    wt_df = csv_all[['mu_wind(m/s)','sigma_wind(m/s)']]
    load_df = csv_all[['hourly_load%']] 
    price_df = csv_all[['price($/MWh)']]
    scaler = StandardScaler() # TODO is this right?

    # Normalize the values for PV and Wind generation -> for later use
    pv_df_scaled = scaler.fit_transform(pv_df)
    wt_df_scaled = scaler.fit_transform(wt_df)
    load_df_scaled = scaler.fit_transform(load_df)
    price_df_scaled = scaler.fit_transform(price_df)
    pv_df_scaled = pd.DataFrame(pv_df_scaled, index=pv_df.index, columns=pv_df.columns)
    wt_df_scaled = pd.DataFrame(wt_df_scaled, index=wt_df.index, columns=wt_df.columns)
    load_df_scaled = pd.DataFrame(load_df_scaled, index=load_df.index, columns=load_df.columns)
    price_df_scaled = pd.DataFrame(price_df_scaled, index=price_df.index, columns=price_df.columns)

    return pv_df,wt_df,load_df,price_df, pv_df_scaled,wt_df_scaled,load_df_scaled,price_df_scaled

def create_save_profile(csv_all):
    pv_df,wt_df,load_df,price_df,pv_df_scaled,wt_df_scaled,load_df_scaled,price_df_scaled = create_unit_profile(csv_all)


    # create csv files
    pv_df_scaled.to_csv('./data/profile/pv_profile.csv')
    wt_df_scaled.to_csv('./data/profile/wt_profile.csv')
    load_df_scaled.to_csv('./data/profile/load_profile.csv')
    price_df_scaled.to_csv('./data/profile/price_profile.csv')

if __name__ == '__main__':
    # Read the solar and wind data CSV generated from the image
    csv_all = pd.read_csv('./data/profile/solar_wind_data.csv')

    # Generate and save profiles
    create_save_profile(csv_all)

    # Load the generated profiles
    pv_df,wt_df,load_df,price_df, pv_df_scaled,wt_df_scaled,load_df_scaled,price_df_scaled = create_unit_profile(csv_all)
    
    # Visualize or perform further analysis
    utils.view_profile(pv_df,wt_df,load_df,price_df)
