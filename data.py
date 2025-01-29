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
    pv_df['s_h_max'] = pv_df['mu_solar(kW/m^2)'] + pv_df['sigma_solar(kW/m^2)']
    pv_df['s_h_min'] = np.maximum(pv_df['mu_solar(kW/m^2)'] - pv_df['sigma_solar(kW/m^2)'],0)

    wt_df = csv_all[['mu_wind(m/s)', 'sigma_wind(m/s)']]
    wt_df['v_h_max'] = wt_df['mu_wind(m/s)'] + wt_df['sigma_wind(m/s)']
    wt_df['v_h_min'] = np.maximum(wt_df['mu_wind(m/s)'] - wt_df['sigma_wind(m/s)'], 0)

    load_df = csv_all[['hourly_load%']]
    load_pl_values = [item[1] for item in load_data]  # Second element is PL (100% load)
    load_df = load_df["hourly_load%"].apply(lambda x: [x * pl / 100 for pl in load_pl_values])
    load_df= pd.DataFrame(load_df.tolist(), columns=[f"C{i}" for i in range(len(load_pl_values))])

    price_df = csv_all[['price($/MWh)']]

    # calculating wind parameters
    wt_df['P_wind_max'] = wt_df['v_h_max'].apply(calculate_wind_power)
    wt_df['P_wind_min'] = wt_df['v_h_min'].apply(calculate_wind_power)

    pv_df['P_solar_max'] = pv_df['s_h_max'].apply(calculate_solar_power) / (10**3)
    pv_df['P_solar_min'] = pv_df['s_h_min'].apply(calculate_solar_power) / (10**3)

    # NORMALIZE DATA
    # pv_df_scaled = normalize_df_column(pv_df, 'P_solar')
    # wt_df_scaled = normalize_df_column(wt_df, 'P_wind')
    # load_df_scaled = normalize_df_column(load_df, load_df.columns)
    # price_df_scaled = normalize_df_column(price_df, 'price($/MWh)')

    # normalized_profiles = pd.concat([pv_df_scaled, wt_df_scaled, load_df_scaled, price_df_scaled], axis=1)
    return pv_df,wt_df,load_df,price_df


def create_save_profile(pv_df,wt_df,load_df,price_df):

    # create csv files
    pv_df.to_csv('./data/derived/pv_profile.csv')
    wt_df.to_csv('./data/derived/wt_profile.csv')
    load_df.to_csv('./data/derived/load_profile.csv')
    price_df.to_csv('./data/derived/price_profile.csv')
    #normalized_profiles.to_csv('./data/derived/normalised.csv')

if __name__ == '__main__':
    csv_all = pd.read_csv('./data/downloaded data/solar_wind_data.csv')
    pv_df,wt_df,load_df,price_df = create_unit_profile(csv_all)
    create_save_profile(pv_df,wt_df,load_df,price_df)    