import pandas as pd
from utils import *
from setting import *
from data import *
def create_unit_profile(csv_all):
    """
    Process and normalize solar, wind, load, and price data from a CSV.
    """
    # Extract relevant dataframes from the input CSV
    pv_df = csv_all[['mu_solar(kW/m^2)', 'sigma_solar(kW/m^2)']]
    #pv_df['s_h'] = solar_wind_speeds['Shortwave Radiation'].iloc[5:29].values
    pv_df['s_h'] = pv_df['mu_solar(kW/m^2)']
    wt_df = csv_all[['mu_wind(m/s)', 'sigma_wind(m/s)']]
    #wt_df['v_h'] = solar_wind_speeds['Wind Speed'].iloc[5:29].values
    wt_df['v_h'] = wt_df['mu_wind(m/s)']
    load_df = csv_all[['hourly_load%']]
    load_pl_values = [item[1] for item in load_data]  # Second element is PL (100% load)
    # Multiply each row in load_df with load_pl_values
    load_df = load_df["hourly_load%"].apply(lambda x: [x * pl / 100 for pl in load_pl_values])

    # Convert the list of scaled loads to a DataFrame
    load_df= pd.DataFrame(load_df.tolist(), columns=[f"C{i}" for i in range(len(load_pl_values))])

    price_df = csv_all[['price($/MWh)']]

    # calculating wind parameters
    #wt_df['k_h_w'], wt_df['c_h_w'] = zip(*wt_df.apply(lambda row: calculate_shape_parameters(row['mu_wind(m/s)'], row['sigma_wind(m/s)']), axis=1))
    #wt_df['f_wind_h'] = wt_df.apply(lambda row: calculate_f_wind(row['v_h'], row['k_h_w'], row['c_h_w']), axis=1)
    wt_df['P_wind'] = wt_df['v_h'].apply(calculate_wind_power)

    # calculating solar parameters

    #pv_df['c_h_s'] = pv_df.apply(lambda row: calculate_chs(row['mu_solar(kW/m^2)'], row['sigma_solar(kW/m^2)']), axis=1)
    #pv_df['k_h_s'] = pv_df.apply(lambda row: calculate_khs(row['mu_solar(kW/m^2)'], row['c_h_s']), axis=1)
    #pv_df['f_h_s'] = pv_df.apply(lambda row: calculate_f_solar(row['s_h'],  row['k_h_s'], row['c_h_s']), axis=1)
    pv_df['P_solar'] = pv_df['s_h'].apply(calculate_solar_power) / (10**3)

    # NORMALIZE DATA
    pv_df_scaled = normalize_df_column(pv_df, 'P_solar')
    wt_df_scaled = normalize_df_column(wt_df, 'P_wind')
    load_df_scaled = normalize_df_column(load_df, load_df.columns)
    price_df_scaled = normalize_df_column(price_df, 'price($/MWh)')

# Concatenate DataFrames along the columns
    normalized_profiles = pd.concat([pv_df_scaled, wt_df_scaled, load_df_scaled, price_df_scaled], axis=1)
    return pv_df,wt_df,load_df,price_df,normalized_profiles


def create_save_profile(pv_df,wt_df,load_df,price_df,normalized_profiles):

    # create csv files
    pv_df.to_csv('./data/derived/pv_profile.csv')
    wt_df.to_csv('./data/derived/wt_profile.csv')
    load_df.to_csv('./data/derived/load_profile.csv')
    price_df.to_csv('./data/derived/price_profile.csv')
    normalized_profiles.to_csv('./data/derived/normalised.csv')

if __name__ == '__main__':
    # Read the solar and wind data CSV generated from the image
    csv_all = pd.read_csv('./data/downloaded data/solar_wind_data.csv')
    pv_df,wt_df,load_df,price_df,normalized_profiles  = create_unit_profile(csv_all)
    # Generate and save profiles
    create_save_profile(pv_df,wt_df,load_df,price_df,normalized_profiles)    