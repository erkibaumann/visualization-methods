# Common imports and helper functions
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import io # To read downloaded content directly into pandas
import json
from datetime import datetime
import warnings # To handle potential UserWarnings during processing more gracefully

# --- Configuration ---
API_URL = "https://decision.cs.taltech.ee/electricity/api/"
DATA_DOWNLOAD_URL_TEMPLATE = "https://decision.cs.taltech.ee/electricity/data/{hash}.csv"

# --- Helper Functions ---

def fetch_dataset_list(api_url):
    """Fetches the list of dataset metadata from the API."""
    print(f"Fetching dataset list from {api_url}")
    try:
        response = requests.get(api_url, timeout=30) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        datasets = response.json()
        print(f"Successfully fetched metadata for {len(datasets)} datasets.")
        return datasets
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset list: {e}")
        return None
    except json.JSONDecodeError:
        print("Error parsing JSON response from API.")
        return None

def download_and_process_csv(dataset_hash, base_url_template):
    """Downloads a specific dataset CSV and preprocesses it."""
    download_url = base_url_template.format(hash=dataset_hash)
    print(f" Downloading and processing: {download_url}")
    try:
        response = requests.get(download_url, timeout=60) # Longer timeout for download
        response.raise_for_status()

        csv_content = io.StringIO(response.text)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(csv_content, sep=';', skiprows=4, header=0)

            if len(df.columns) == 2:
                df.columns = ['Periood', 'consumption']
            else:
                print(f"    Warning: Expected 2 columns, found {len(df.columns)} for {dataset_hash}. Skipping.")
                return None

            df['Periood'] = pd.to_datetime(df['Periood'], dayfirst=True, errors='coerce')
            df['consumption'] = pd.to_numeric(df['consumption'].astype(str).str.replace(',', '.', regex=False), errors='coerce')

        df.dropna(subset=['Periood', 'consumption'], inplace=True)

        df['date'] = df['Periood'].dt.date
        df['hour'] = df['Periood'].dt.hour
        df['dataset_hash'] = dataset_hash

        unique_days_count = df['date'].nunique()
        if unique_days_count > 100:
            print(f"  Successfully processed {dataset_hash}. Shape: {df.shape}, Unique days: {unique_days_count}")
            return df
        else:
            print(f"    Warning: Only {unique_days_count} unique days found for {dataset_hash} (less than 100). Processing anyway.")
            if unique_days_count == 0:
                 print(f"    Error: No valid date entries found for {dataset_hash}. Skipping.")
                 return None
            return df

    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {dataset_hash}: {e}")
        return None
    except ValueError as e:
         print(f"  Error converting data types for {dataset_hash}: {e}")
         return None
    except Exception as e:
        print(f"  An unexpected error occurred processing {dataset_hash}: {e}")
        return None

# --- Task 1 Implementation ---

def run_task1(datasets, base_url_template, dataset_index=0):
    """Generates heatmap for 100 days of a single chosen dataset."""
    print("\n--- Running Task 1 ---")
    if not datasets:
        print("No datasets available to process.")
        return

    if dataset_index >= len(datasets):
        print(f"Error: dataset_index {dataset_index} is out of bounds (0-{len(datasets)-1}). Using index 0.")
        dataset_index = 0

    chosen_dataset_info = datasets[dataset_index]
    chosen_hash = chosen_dataset_info['dataset']
    print(f"Selected dataset {dataset_index+1} / {len(datasets)}: {chosen_hash}")

    df_single_meter = download_and_process_csv(chosen_hash, base_url_template)

    if df_single_meter is not None and not df_single_meter.empty:
        df_single_meter.sort_values('Periood', inplace=True)
        unique_dates = sorted(df_single_meter['date'].unique())
        if len(unique_dates) == 0:
            print(f"Error: No valid dates found after processing {chosen_hash}. Cannot proceed.")
            return

        if len(unique_dates) < 100:
             print(f"Warning: Only {len(unique_dates)} unique days available. Visualizing all available days.")
             dates_to_plot = unique_dates
        else:
             dates_to_plot = unique_dates[:100]

        start_date_obj = dates_to_plot[0]
        end_date_obj = dates_to_plot[-1]
        start_date_str_eu = start_date_obj.strftime('%d.%m.%Y')
        end_date_str_eu = end_date_obj.strftime('%d.%m.%Y')

        df_plot_period = df_single_meter[df_single_meter['date'].isin(dates_to_plot)].copy()
        print(f"Selected data for {chosen_hash} from {start_date_str_eu} to {end_date_str_eu}. Shape: {df_plot_period.shape}")

        if not df_plot_period.empty:
            try:
                heatmap_data = df_plot_period.pivot_table(
                    index='date', columns='hour', values='consumption', aggfunc='mean'
                )
                heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=np.nan)
                heatmap_data.sort_index(inplace=True)

                valid_data = heatmap_data.values[~np.isnan(heatmap_data.values)]
                if len(valid_data) > 0:
                    color_min = np.nanpercentile(heatmap_data.values, 1)
                    color_max = np.nanpercentile(heatmap_data.values, 99) # 99th percentile
                    if color_min == color_max:
                         color_min = color_min - 0.5 if color_min > 0 else color_min - (color_min * 0.1)
                         color_max = color_max + 0.5 if color_max > 0 else color_max + (color_max * 0.1)
                         if color_min == color_max:
                             color_min -= 1
                             color_max += 1
                    print(f"  Color scale range (1st-99th percentile): {color_min:.2f} to {color_max:.2f}")
                else:
                    print("  Warning: No valid numeric data found for color scaling.")
                    color_min, color_max = None, None

                heatmap_data.index = pd.to_datetime(heatmap_data.index)
                y_axis_labels = heatmap_data.index.strftime('%d.%m.%Y')

                num_rows = len(heatmap_data.index)
                num_cols = len(heatmap_data.columns)
                plot_height = max(1000, num_rows * 30)
                plot_width = min(1200, int(num_cols * (plot_height / num_rows) * 1.8)) # Adjusted width slightly
                print(f"  Plot dimensions: Height={plot_height}px, Width={plot_width}px")

                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Hour of Day", y="Date", color="Consumption"),
                    x=[str(h) for h in range(num_cols)],
                    y=y_axis_labels,
                    title=f'Hourly Consumption Heatmap ({chosen_hash})<br>{start_date_str_eu} - {end_date_str_eu} ({len(dates_to_plot)} Days)',
                    aspect="auto",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    zmin=color_min, zmax=color_max
                )
                fig.update_xaxes(side="top", dtick=1)
                tick_interval = max(1, len(dates_to_plot) // 14)
                fig.update_yaxes(dtick=tick_interval)
                fig.update_layout(
                    yaxis_title="Date", xaxis_title="Hour",
                    height=plot_height, width=plot_width,
                    coloraxis_colorbar=dict(title="Consumption")
                )

                # --- Save Plot with fixed name ---
                output_file = 'task1_visualization.html' # MODIFIED
                fig.write_html(output_file)
                print(f"Task 1: Interactive heatmap saved to {output_file}")

            except Exception as e:
                print(f"Error creating pivot table or heatmap for Task 1 ({chosen_hash}): {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"No data within the selected date range for {chosen_hash}.")
    else:
        print(f"Failed to process data for the selected dataset: {chosen_hash}")

# --- Task 2 Implementation ---

def run_task2(datasets, base_url_template, target_date_str_iso, max_datasets_to_plot=75):
    print("\n--- Running Task 2 ---")
    if not datasets:
        print("No datasets available to process.")
        return

    try:
        target_date_obj = datetime.strptime(target_date_str_iso, '%Y-%m-%d').date()
        target_date_str_eu = target_date_obj.strftime('%d.%m.%Y')
        print(f"Target date: {target_date_str_eu} (parsed from {target_date_str_iso})")
    except ValueError:
        print(f"Error: Invalid target_date format '{target_date_str_iso}'. Please use YYYY-MM-DD.")
        return

    all_sameday_data = []
    datasets_to_consider = datasets[:min(len(datasets), max_datasets_to_plot + 20)]
    print(f"Attempting to process up to {len(datasets_to_consider)} datasets for {target_date_str_eu}...")
    processed_hashes = set()

    for dataset_info in datasets_to_consider:
        if len(processed_hashes) >= max_datasets_to_plot:
             break
        dataset_hash = dataset_info['dataset']
        if dataset_hash in processed_hashes:
            continue

        df_full = download_and_process_csv(dataset_hash, base_url_template)
        if df_full is not None and not df_full.empty:
            df_target_day = df_full[df_full['date'] == target_date_obj].copy()
            if not df_target_day.empty:
                if len(df_target_day['hour'].unique()) == 24:
                    all_sameday_data.append(df_target_day[['dataset_hash', 'hour', 'consumption']])
                    processed_hashes.add(dataset_hash)
                    print(f"  Data found for {target_date_str_eu} in {dataset_hash}. Count: {len(processed_hashes)}")
                else:
                    print(f"  Skipping {dataset_hash}: Incomplete data ({len(df_target_day['hour'].unique())}/24 hours).")

    processed_count = len(processed_hashes)
    if processed_count == 0:
        print(f"No complete datasets found containing data for the target date: {target_date_str_eu}")
        return

    df_combined = pd.concat(all_sameday_data, ignore_index=True)
    print(f"Combined data for {target_date_str_eu} from {processed_count} datasets. Shape: {df_combined.shape}")

    try:
        heatmap_data_sameday = df_combined.pivot_table(
            index='dataset_hash', columns='hour', values='consumption', aggfunc='mean'
        )
        heatmap_data_sameday = heatmap_data_sameday.reindex(columns=range(24), fill_value=np.nan)
        heatmap_data_sameday['total_consumption'] = heatmap_data_sameday.sum(axis=1)
        heatmap_data_sameday.sort_values('total_consumption', ascending=False, inplace=True)
        heatmap_data_sameday.drop(columns=['total_consumption'], inplace=True)

        valid_data_sameday = heatmap_data_sameday.values[~np.isnan(heatmap_data_sameday.values)]
        if len(valid_data_sameday) > 0:
            color_min_sameday = np.nanpercentile(heatmap_data_sameday.values, 1)
            color_max_sameday = np.nanpercentile(heatmap_data_sameday.values, 95) # MODIFIED from 75 to 95
            if color_min_sameday == color_max_sameday:
                 color_min_sameday = color_min_sameday - 0.5 if color_min_sameday > 0 else color_min_sameday - (color_min_sameday * 0.1)
                 color_max_sameday = color_max_sameday + 0.5 if color_max_sameday > 0 else color_max_sameday + (color_max_sameday * 0.1)
                 if color_min_sameday == color_max_sameday:
                     color_min_sameday -= 1
                     color_max_sameday += 1
            print(f"  Color scale range (1st-95th percentile): {color_min_sameday:.2f} to {color_max_sameday:.2f}")
        else:
             print("  Warning: No valid numeric data found for color scaling.")
             color_min_sameday, color_max_sameday = None, None

        num_rows_sameday = len(heatmap_data_sameday.index)
        num_cols_sameday = len(heatmap_data_sameday.columns)
        plot_height_sameday = max(1000, num_rows_sameday * 15) # Adjusted height slightly
        plot_width_sameday = min(1200, int(num_cols_sameday * (plot_height_sameday / num_rows_sameday) * 1.5)) # Adjusted width slightly
        print(f"  Plot dimensions: Height={plot_height_sameday}px, Width={plot_width_sameday}px")

        fig_sameday = px.imshow(
            heatmap_data_sameday,
            labels=dict(x="Hour of Day", y="Dataset Hash", color="Consumption"),
            x=[str(h) for h in range(num_cols_sameday)],
            y=heatmap_data_sameday.index,
            title=f'Hourly Consumption Comparison for {target_date_str_eu} ({processed_count} Datasets)',
            aspect="auto",
            color_continuous_scale=px.colors.sequential.Viridis,
            zmin=color_min_sameday, zmax=color_max_sameday
        )
        fig_sameday.update_xaxes(side="top", dtick=1)
        if processed_count > 60:
             fig_sameday.update_yaxes(showticklabels=False, title="Dataset Hash (Hover to see)")
        else:
             fig_sameday.update_yaxes(title="Dataset Hash", dtick=1)

        fig_sameday.update_layout(
             height=plot_height_sameday, width=plot_width_sameday,
             coloraxis_colorbar=dict(title="Consumption")
        )

        # --- Save Plot with fixed name ---
        output_file = 'task2_visualization.html' # MODIFIED
        fig_sameday.write_html(output_file)
        print(f"Task 2: Interactive heatmap saved to {output_file}")

    except Exception as e:
         print(f"Error creating pivot table or heatmap for Task 2 ({target_date_str_eu}): {e}")
         import traceback
         traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    all_datasets = fetch_dataset_list(API_URL)

    if all_datasets:
        # --- Run Task 1 ---
        # Choose which dataset to visualize (e.g., index 13 for the 14th dataset)
        run_task1(all_datasets, DATA_DOWNLOAD_URL_TEMPLATE, dataset_index=13)

        # --- Run Task 2 ---
        # Choose a target date in YYYY-MM-DD format.
        # Ensure this date has data in the datasets. Example: "2023-03-15"
        target_date_for_task2 = "2023-03-15"
        max_datasets_compare = 75
        run_task2(all_datasets, DATA_DOWNLOAD_URL_TEMPLATE, target_date_for_task2, max_datasets_to_plot=max_datasets_compare)
    else:
        print("Could not retrieve dataset list. Exiting.")