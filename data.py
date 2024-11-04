import pandas as pd
from utils import *
from setting import *
def create_unit_profile(csv_all):
    """
    Process and normalize solar, wind, load, and price data from a CSV.
    """
    # Extract relevant dataframes from the input CSV
    pv_df = csv_all[['mu_solar(kW/m^2)', 'sigma_solar(kW/m^2)']]
    wt_df = csv_all[['mu_wind(m/s)', 'sigma_wind(m/s)']]
    load_df = csv_all[['hourly_load%']]
    price_df = csv_all[['price($/MWh)']]

    # calculating wind and solar power outputs
    wind_speedpdf = pd.DataFrame(wt_df)
    wind_speedpdf['k_h_w'], wind_speedpdf['c_h_w'] = zip(*wind_speedpdf.apply(lambda row: calculate_shape_parameters(row['mu_wind(m/s)'], row['sigma_wind(m/s)']), axis=1))
    wind_speedpdf['f_wind_h'] = wind_speedpdf.apply(lambda row: calculate_f_wind(v_h, row['mu_h_wind'], row['sigma_h_wind']), axis=1)
    
    # TODO change this for wind speed instead of mu_wind
    # Normalize the wind data
    wind_min = np.min(wt_df['mu_wind(m/s)'])
    wind_max = np.max(wt_df['mu_wind(m/s)'])
    wt_df_scaled = (wt_df['mu_wind(m/s)'] - wind_min) / (wind_max - wind_min)

    # Normalize the solar data
    solar_min = np.min(pv_df['mu_solar(kW/m^2)'])
    solar_max = np.max(pv_df['mu_solar(kW/m^2)'])
    pv_df_scaled = (pv_df['mu_solar(kW/m^2)'] - solar_min) / (solar_max - solar_min)

    # Normalize the load data
    load_min = np.min(load_df['hourly_load%'])
    load_max = np.max(load_df['hourly_load%'])
    load_df_scaled = (load_df['hourly_load%'] - load_min) / (load_max - load_min)

    # Normalize the price data
    price_min = np.min(price_df['price($/MWh)'])
    price_max = np.max(price_df['price($/MWh)'])
    price_df_scaled = (price_df['price($/MWh)'] - price_min) / (price_max - price_min)

    # Combine normalized data into a single dictionary or DataFrame if needed
    normalized_profiles = {
        'pv_df_scaled': pv_df_scaled,
        'wt_df_scaled': wt_df_scaled,
        'load_df_scaled': load_df_scaled,
        'price_df_scaled': price_df_scaled
    }

    return pv_df,wt_df,load_df,price_df,normalized_profiles


def create_save_profile(csv_all):
    all = create_unit_profile(csv_all)


    # create csv files
    all['pv_df_scaled'].to_csv('./data/profile/pv_profile.csv')
    all['wt_df_scaled'].to_csv('./data/profile/wt_profile.csv')
    all['load_df_scaled'].to_csv('./data/profile/load_profile.csv')
    all['price_df_scaled'].to_csv('./data/profile/price_profile.csv')

if __name__ == '__main__':
    # Read the solar and wind data CSV generated from the image
    csv_all = pd.read_csv('./data/profile/solar_wind_data.csv')

    # Generate and save profiles
    create_save_profile(csv_all)

    # Load the generated profiles
    pv_df,wt_df,load_df,price_df, _ = create_unit_profile(csv_all)
    
    # Visualize or perform further analysis
    utils.view_profile(pv_df,wt_df,load_df,price_df)
