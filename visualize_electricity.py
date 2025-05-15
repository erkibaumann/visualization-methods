# Common imports and helper functions
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io # To read downloaded content directly into pandas
import json
from datetime import datetime, date, timedelta
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


def run_task1(datasets, base_url_template, dataset_index=0):
    """
    Generates a horizontally long heatmap with a weekday/weekend annotation track.
    Days on X-axis, hours on Y-axis.
    """
    print("\n--- Running Task 1 (Horizontal Layout with Weekday Annotation Track) ---")
    if not datasets:
        print("No datasets available to process.")
        return

    if not datasets or dataset_index < 0 or dataset_index >= len(datasets):
        actual_index = 0
        if datasets:
            actual_index = dataset_index % len(datasets) if len(datasets) > 0 else 0
        print(
            f"Warning: dataset_index {dataset_index} is out of bounds (0-{len(datasets) - 1}). Using index {actual_index}.")
        dataset_index = actual_index

    chosen_dataset_info = datasets[dataset_index]
    chosen_hash = chosen_dataset_info['dataset']
    print(f"Selected dataset {dataset_index + 1} / {len(datasets)}: {chosen_hash}")

    df_single_meter = download_and_process_csv(chosen_hash, base_url_template)

    if df_single_meter is not None and not df_single_meter.empty:
        df_single_meter.sort_values('Periood', inplace=True)

        if not df_single_meter['date'].empty:
            first_date_element = df_single_meter['date'].iloc[0]
            if not isinstance(first_date_element, date) or isinstance(first_date_element, pd.Timestamp):
                try:
                    df_single_meter['date'] = pd.to_datetime(df_single_meter['date'], errors='coerce').dt.date
                    df_single_meter.dropna(subset=['date'], inplace=True)
                except Exception as e:
                    print(f"Warning: Could not convert 'date' column to date objects for {chosen_hash}. Error: {e}")

        unique_dates_dt = sorted(df_single_meter['date'].dropna().unique())

        if not unique_dates_dt:
            print(f"Error: No valid dates found after processing {chosen_hash}. Cannot proceed.")
            return

        num_available_days = len(unique_dates_dt)
        dates_to_plot_dt = unique_dates_dt[:100] if num_available_days >= 100 else unique_dates_dt
        if num_available_days < 100:
            print(f"Warning: Only {num_available_days} unique days available. Visualizing all available days.")

        if not dates_to_plot_dt:
            print(f"Error: No dates selected for plotting for {chosen_hash}.")
            return

        start_date_obj = dates_to_plot_dt[0]
        end_date_obj = dates_to_plot_dt[-1]
        if not (isinstance(start_date_obj, date) and isinstance(end_date_obj, date)):
            print(f"Error: start_date_obj or end_date_obj is not a valid date object for {chosen_hash}.")
            return

        start_date_str_eu = start_date_obj.strftime('%d.%m.%Y')
        end_date_str_eu = end_date_obj.strftime('%d.%m.%Y')

        df_plot_period = df_single_meter[df_single_meter['date'].isin(dates_to_plot_dt)].copy()
        print(
            f"Selected data for {chosen_hash} from {start_date_str_eu} to {end_date_str_eu}. Shape: {df_plot_period.shape}")

        if not df_plot_period.empty:
            try:
                heatmap_data_original = df_plot_period.pivot_table(
                    index='date', columns='hour', values='consumption', aggfunc='mean'
                )
                heatmap_data_original = heatmap_data_original.reindex(columns=range(24), fill_value=np.nan)
                heatmap_data_original.sort_index(inplace=True)
                heatmap_data_transposed = heatmap_data_original.T

                valid_data = heatmap_data_transposed.values[~np.isnan(heatmap_data_transposed.values)]
                if len(valid_data) > 0:
                    color_min = np.nanpercentile(valid_data, 10)
                    color_max = np.nanpercentile(valid_data, 90)
                    if color_min == color_max:
                        delta = abs(color_min * 0.1) if color_min != 0 else 0.1
                        color_min = color_min - delta if delta > 0 else color_min - 1
                        color_max = color_max + delta if delta > 0 else color_max + 1
                        if color_min == color_max: color_min -= 1; color_max += 1  # Final fallback
                    print(f"  Color scale range (1st-80th percentile): {color_min:.2f} to {color_max:.2f}")
                else:
                    print("  Warning: No valid numeric data found for color scaling.")
                    color_min, color_max = None, None

                # --- Prepare X and Y axis labels for main heatmap ---
                # X-axis: Dates (columns of transposed data)
                # Convert date columns to Pandas Timestamps for strftime, then to strings
                x_axis_dates_obj = pd.to_datetime(heatmap_data_transposed.columns).date  # Get as python date objects
                x_axis_labels_str = [d.strftime('%d.%m.%y') for d in x_axis_dates_obj]

                # Y-axis: Hours (index of transposed data)
                y_axis_labels_str = [str(h) for h in heatmap_data_transposed.index]

                num_days_on_x = len(x_axis_labels_str)

                # --- Prepare data for Weekday/Weekend Annotation Track ---
                day_type_numeric = []
                day_type_hover_text = []
                for dt_obj in x_axis_dates_obj:  # these are now python date objects
                    day_of_week = dt_obj.weekday()  # Monday=0, Sunday=6
                    if day_of_week == 5:  # Saturday
                        day_type_numeric.append(1)
                        day_type_hover_text.append(f"{dt_obj.strftime('%d.%m.%y')}<br>Laupäev")
                    elif day_of_week == 6:  # Sunday
                        day_type_numeric.append(2)
                        day_type_hover_text.append(f"{dt_obj.strftime('%d.%m.%y')}<br>Pühapäev")
                    else:  # Weekday
                        day_type_numeric.append(0)
                        day_type_hover_text.append(f"{dt_obj.strftime('%d.%m.%y')}<br>Tööpäev")

                z_annotation = [day_type_numeric]  # Must be 2D for heatmap's z
                y_annotation_label = ['Päeva tüüp']

                annotation_colors = ['rgb(220,220,220)', 'rgb(173,216,230)',
                                     'rgb(100,149,237)']  # Grey, LightBlue, CornflowerBlue
                annotation_colorscale = [
                    [0.0, annotation_colors[0]],  # for value 0 (Weekday)
                    [0.5, annotation_colors[1]],  # for value 1 (Saturday)
                    [1.0, annotation_colors[2]]  # for value 2 (Sunday)
                ]

                # --- Initialize Figure ---
                fig = go.Figure()

                # --- Add Main Consumption Heatmap Trace ---
                fig.add_trace(go.Heatmap(
                    z=heatmap_data_transposed.values,
                    x=x_axis_labels_str,
                    y=y_axis_labels_str,
                    colorscale='Plasma',
                    zmin=color_min,
                    zmax=color_max,
                    colorbar=dict(title="Tarbimine (kWh)", yanchor="bottom", y=0.07, len=0.93),
                    # Adjust colorbar position
                    hoverongaps=False,
                    name="Tarbimine",
                    customdata=heatmap_data_transposed.T.stack().reset_index(name='consumption')['consumption'],
                    # For hover
                    hovertemplate="Kuupäev: %{x}<br>Tund: %{y}<br>Tarbimine: %{z:.2f} kWh<extra></extra>",
                    yaxis='y1'  # Assign to the first y-axis
                ))

                # --- Add Weekday/Weekend Annotation Heatmap Trace ---
                fig.add_trace(go.Heatmap(
                    z=z_annotation,
                    x=x_axis_labels_str,
                    y=y_annotation_label,
                    colorscale=annotation_colorscale,
                    showscale=False,  # No separate colorbar for this
                    hoverinfo='text',  # Use custom text for hover
                    text=[day_type_hover_text],  # Custom text for hover (must be 2D)
                    name="Päeva tüüp",
                    yaxis='y2'  # Assign to the second y-axis
                ))

                # --- Plot Dimensions ---
                num_hours_on_y = len(y_axis_labels_str)
                cell_height_px = 18
                cell_width_px = 10

                title_height = 80
                xaxis_label_height = 70  # Space needed for x-axis labels and title
                annotation_track_visual_height = 25
                gap_between_tracks = 10

                main_heatmap_content_height = num_hours_on_y * cell_height_px

                # The total plot height needs to accommodate all these PLUS the x-axis labels at the very bottom
                plot_height_total = (main_heatmap_content_height +
                                     annotation_track_visual_height +
                                     gap_between_tracks +
                                     title_height +
                                     xaxis_label_height)  # xaxis_label_height is now for the space at the bottom

                plot_width_total = num_days_on_x * cell_width_px + 150

                plot_height_total = max(500, min(int(plot_height_total), 900))
                plot_width_total = max(1000, min(int(plot_width_total), 3500))

                print(f"  Plot dimensions: Height={plot_height_total}px, Width={plot_width_total}px")

                # --- Define Y-Axis Domains ---
                # Content area is height excluding title and bottom x-axis space
                content_area_for_y_axes = plot_height_total - title_height - xaxis_label_height

                # Calculate domain heights relative to this content_area_for_y_axes
                # Domain values are [bottom_edge, top_edge] as fractions of content_area_for_y_axes

                # Annotation track (y2) at the bottom of the y-axis content area
                y2_bottom_frac = 0.0
                y2_top_frac = annotation_track_visual_height / content_area_for_y_axes

                # Main heatmap (y1) above y2, with a gap
                y1_bottom_frac = (annotation_track_visual_height + gap_between_tracks) / content_area_for_y_axes
                y1_top_frac = 1.0  # Takes remaining space up to the top of y-axis content area

                # Ensure fractions are valid and make sense
                y2_top_frac = min(y2_top_frac, 0.15)  # Cap annotation track height share
                y1_bottom_frac = min(y1_bottom_frac, y2_top_frac + 0.05)  # Ensure gap is reasonable
                if y1_bottom_frac >= y1_top_frac:  # Safety if plot is too short or calcs are off
                    y2_top_frac = 0.1
                    y1_bottom_frac = 0.15
                    y1_top_frac = 1.0

                # Now, convert these fractions relative to content_area_for_y_axes
                # back to domains relative to the *overall plot height* (0 to 1 range for layout.yaxis.domain)
                # The y-axes domains start from where the bottom margin (xaxis_label_height) ends,
                # and go up to where the top margin (title_height) begins.

                bottom_plot_margin_frac = xaxis_label_height / plot_height_total
                top_plot_margin_frac = (plot_height_total - title_height) / plot_height_total

                yaxis2_domain_bottom = bottom_plot_margin_frac + (
                            y2_bottom_frac * (top_plot_margin_frac - bottom_plot_margin_frac))
                yaxis2_domain_top = bottom_plot_margin_frac + (
                            y2_top_frac * (top_plot_margin_frac - bottom_plot_margin_frac))

                yaxis1_domain_bottom = bottom_plot_margin_frac + (
                            y1_bottom_frac * (top_plot_margin_frac - bottom_plot_margin_frac))
                yaxis1_domain_top = bottom_plot_margin_frac + (
                            y1_top_frac * (top_plot_margin_frac - bottom_plot_margin_frac))

                yaxis2_domain = [yaxis2_domain_bottom, yaxis2_domain_top]
                yaxis1_domain = [yaxis1_domain_bottom, yaxis1_domain_top]

                # --- Update Layout ---
                num_x_ticks_to_show = 20
                x_tick_interval = max(1, num_days_on_x // num_x_ticks_to_show) if num_days_on_x > 0 else 1

                fig.update_layout(
                    title_text=f'Tunnipõhine elektritarbimine: {chosen_hash}<br>{start_date_str_eu} - {end_date_str_eu} ({num_days_on_x} päeva)',
                    title_x=0.5,
                    xaxis=dict(
                        side="bottom",
                        dtick=x_tick_interval,
                        tickangle=30,
                        title_text="Kuupäev",
                        anchor='y2',  # <--- ANCHOR X-AXIS TO THE BOTTOM OF YAXIS2 (annotation track)
                        # The position of x-axis labels will be handled by the bottom margin
                    ),
                    yaxis=dict(  # Main consumption heatmap y-axis (y1)
                        title="Tund",
                        dtick=1,
                        autorange="reversed",
                        domain=yaxis1_domain,
                    ),
                    yaxis2=dict(  # Annotation track y-axis (y2)
                        showticklabels=False,
                        ticks="",
                        domain=yaxis2_domain,
                        fixedrange=True
                    ),
                    height=plot_height_total,
                    width=plot_width_total,
                    margin=dict(l=70, r=50, b=xaxis_label_height, t=title_height, pad=4),
                    # Ensure bottom margin is respected
                    hovermode='closest'
                )

                output_file = 'task1_visualization.html'
                fig.write_html(output_file)
                print(f"Task 1: Interactive heatmap with annotation track saved to {output_file}")

            except Exception as e:
                print(f"Error creating pivot table or heatmap for Task 1 ({chosen_hash}): {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"No data within the selected date range for {chosen_hash}.")
    else:
        print(f"Failed to process data for the selected dataset: {chosen_hash}")


# --- Task 2 Implementation (Modified) ---

def run_task2(datasets, base_url_template, target_date_str_iso, max_datasets_to_plot=75, normalize_rows=True): # Added normalize_rows flag
    print(f"\n--- Running Task 2 (Normalize Rows: {normalize_rows}) ---")
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
    ordered_processed_hashes = []
    datasets_to_consider = datasets[:min(len(datasets), max_datasets_to_plot + 30)]
    print(f"Attempting to process up to {len(datasets_to_consider)} datasets for {target_date_str_eu}...")
    processed_hashes_set = set()

    for i, dataset_info in enumerate(datasets_to_consider):
        if len(ordered_processed_hashes) >= max_datasets_to_plot:
            print(f"Reached max_datasets_to_plot ({max_datasets_to_plot}). Stopping data collection.")
            break
        dataset_hash = dataset_info['dataset']
        if dataset_hash in processed_hashes_set:
            continue
        df_full = download_and_process_csv(dataset_hash, base_url_template)
        if df_full is not None and not df_full.empty:
            if not df_full['date'].empty and not isinstance(df_full['date'].iloc[0], date):
                df_full['date'] = pd.to_datetime(df_full['date']).dt.date
            df_target_day = df_full[df_full['date'] == target_date_obj].copy()
            if not df_target_day.empty:
                if len(df_target_day['hour'].unique()) == 24:
                    all_sameday_data.append(df_target_day[['dataset_hash', 'hour', 'consumption']])
                    ordered_processed_hashes.append(dataset_hash)
                    processed_hashes_set.add(dataset_hash)
                    if len(ordered_processed_hashes) % 10 == 0 or len(ordered_processed_hashes) == max_datasets_to_plot:
                        print(
                            f"    Successfully added data for {dataset_hash}. Total datasets for target date: {len(ordered_processed_hashes)}")

    processed_count = len(ordered_processed_hashes)
    if processed_count == 0:
        print(f"No complete datasets found containing data for the target date: {target_date_str_eu}")
        return

    df_combined = pd.concat(all_sameday_data, ignore_index=True)
    print(f"Combined data for {target_date_str_eu} from {processed_count} datasets. Shape: {df_combined.shape}")

    try:
        heatmap_data_sameday_abs = df_combined.pivot_table(
            index='dataset_hash', columns='hour', values='consumption', aggfunc='mean'
        )
        heatmap_data_sameday_abs = heatmap_data_sameday_abs.reindex(columns=range(24), fill_value=np.nan)

        if ordered_processed_hashes:
            valid_ordered_hashes = [h for h in ordered_processed_hashes if h in heatmap_data_sameday_abs.index]
            heatmap_data_sameday_abs = heatmap_data_sameday_abs.reindex(valid_ordered_hashes)

        # --- ROW-WISE NORMALIZATION ---
        if normalize_rows:
            print("  Applying row-wise Min-Max normalization...")
            # Apply Min-Max scaling row by row
            # Create a copy to store normalized data
            heatmap_data_sameday_normalized = heatmap_data_sameday_abs.copy()
            for index, row in heatmap_data_sameday_normalized.iterrows():
                row_min = row.min()
                row_max = row.max()
                if row_max == row_min:  # Avoid division by zero if all values in a row are the same
                    heatmap_data_sameday_normalized.loc[index] = 0.5  # Or 0, or 1, depending on preference
                else:
                    heatmap_data_sameday_normalized.loc[index] = (row - row_min) / (row_max - row_min)

            data_to_plot = heatmap_data_sameday_normalized
            colorbar_title = "Normaliseeritud Tarbimine (0-1)"
            # For normalized data (0-1), zmin and zmax are typically fixed
            color_min_plot = 0.0
            color_max_plot = 1.0
            print(f"  Color scale for normalized data: {color_min_plot:.2f} to {color_max_plot:.2f}")
        else:
            print("  Using absolute consumption values...")
            data_to_plot = heatmap_data_sameday_abs
            colorbar_title = "Tarbimine (kWh)"
            # Use percentile clipping for absolute values
            valid_data_sameday = data_to_plot.values[~np.isnan(data_to_plot.values)]
            if len(valid_data_sameday) > 0:
                color_min_plot = np.nanpercentile(valid_data_sameday, 10)
                color_max_plot = np.nanpercentile(valid_data_sameday, 90)
                if color_min_plot == color_max_plot:
                    delta = abs(color_min_plot * 0.1) if color_min_plot != 0 else 0.1
                    color_min_plot = color_min_plot - delta if delta > 0 else color_min_plot - 1
                    color_max_plot = color_max_plot + delta if delta > 0 else color_max_plot + 1
                    if color_min_plot == color_max_plot: color_min_plot -= 1; color_max_plot += 1
                print(
                    f"  Color scale for absolute data (10th-90th percentile): {color_min_plot:.2f} to {color_max_plot:.2f}")
            else:
                print("  Warning: No valid numeric data found for color scaling.")
                color_min_plot, color_max_plot = None, None

        # --- Plot Dimensions (remains the same) ---
        num_rows_sameday = len(data_to_plot.index)
        num_cols_sameday = 24
        cell_height_px_sameday = 18
        cell_width_px_sameday = 30
        padding_vertical_sameday = 150
        padding_horizontal_sameday = 150
        plot_height_sameday = num_rows_sameday * cell_height_px_sameday + padding_vertical_sameday
        plot_width_sameday = num_cols_sameday * cell_width_px_sameday + padding_horizontal_sameday
        plot_height_sameday = max(600, min(int(plot_height_sameday), 3500))
        plot_width_sameday = max(800, min(int(plot_width_sameday), 1400))
        print(f"  Plot dimensions: Height={plot_height_sameday}px, Width={plot_width_sameday}px")

        # --- Create Interactive Heatmap ---
        fig_sameday = px.imshow(
            data_to_plot,  # Use either normalized or absolute data
            labels=dict(x="Tund", y="Andmestiku ID", color=colorbar_title),  # Dynamic colorbar title
            x=[str(h) for h in range(num_cols_sameday)],
            y=data_to_plot.index,
            title=f'Tunnipõhine tarbimise võrdlus kuupäeval {target_date_str_eu} ({processed_count} andmestikku)',
            aspect="auto",
            color_continuous_scale=px.colors.sequential.Plasma,
            zmin=color_min_plot,  # Use calculated min for color scale
            zmax=color_max_plot  # Use calculated max for color scale
        )

        fig_sameday.update_xaxes(side="top", dtick=1, title_text="Tund", tickangle=0)

        if processed_count > 70:
            fig_sameday.update_yaxes(showticklabels=False, title_text="Andmestiku ID (näha hiirega)")
        else:
            fig_sameday.update_yaxes(title_text="Andmestiku ID", dtick=1)

        fig_sameday.update_layout(
            height=plot_height_sameday,
            width=plot_width_sameday,
            coloraxis_colorbar=dict(
                title=colorbar_title,  # Dynamic colorbar title
                len=0.6,
                y=0.5,
                yanchor='middle'
            ),
            margin=dict(l=100, r=50, b=70, t=100, pad=4)
        )

        # --- Add hover information for original absolute values if normalized ---
        if normalize_rows:
            # We need to provide the original absolute values for hover
            # `customdata` in px.imshow is a bit tricky with direct DataFrame input for this.
            # A common workaround is to build the hover text manually if needed,
            # or accept that hover shows normalized values.
            # For simplicity here, hover will show normalized values.
            # To show absolute values, you'd typically switch to go.Heatmap for more control over hover.
            fig_sameday.update_traces(
                hovertemplate="Andmestik: %{y}<br>Tund: %{x}<br>Normaliseeritud väärtus: %{z:.2f}<extra></extra>")
        else:
            fig_sameday.update_traces(
                hovertemplate="Andmestik: %{y}<br>Tund: %{x}<br>Tarbimine: %{z:.2f} kWh<extra></extra>")

        output_file = 'task2_visualization.html'
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
        # run_task1(all_datasets, DATA_DOWNLOAD_URL_TEMPLATE, dataset_index=13)

        # --- Run Task 2 ---
        # Choose a target date in YYYY-MM-DD format.
        # Ensure this date has data in the datasets. Example: "2023-03-15"
        target_date_for_task2 = "2023-03-15"
        max_datasets_compare = 75
        run_task2(all_datasets, DATA_DOWNLOAD_URL_TEMPLATE, target_date_for_task2, max_datasets_to_plot=max_datasets_compare, normalize_rows=True)
    else:
        print("Could not retrieve dataset list. Exiting.")