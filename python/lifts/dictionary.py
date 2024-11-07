import pandas as pd


def timetable(terminal):
    df = pd.read_csv('C:/Users/Irena Tong/PycharmProjects/simulation_test/data/train_consist_plan.csv')

    df_terminal = df[(df['Destination_ID'] == terminal) & (df['Train_Type'] == 'Intermodal')]

    df_grouped = df_terminal.groupby('Train_ID').agg({
        'Cars_Loaded': find_duplicate_numbers,
        'Cars_Empty': find_duplicate_numbers,
        'Arrival_Time_Actual_Hr': 'min',
        'Departure_Time_Actual_Hr': 'min'
    }).reset_index()

    df_grouped_sorted = df_grouped.sort_values(by='Arrival_Time_Actual_Hr', ascending=True)

    train_dict_list = convert_df_to_dict(df_grouped_sorted)

    return train_dict_list


def convert_df_to_dict(df):
    train_timetable = []
    for _, row in df.iterrows():
        train_dict = {
            "train_id": row['Train_ID'],
            "arrival_time": row['Departure_Time_Actual_Hr'],
            "departure_time": row['Arrival_Time_Actual_Hr'],
            "empty_cars": row['Cars_Empty'],
            "full_cars": row['Cars_Loaded'],
            "oc_number": calculate_oc_number(row),
            "truck_number": calculate_truck_number(row)
        }
        train_timetable.append(train_dict)
    return train_timetable


def truck_resource(data):
    total_truck_number = sum(item['truck_number'] for item in data)
    return total_truck_number


def calculate_oc_number(row):
    return 40

def calculate_truck_number(row):
    full_cars = row['Cars_Loaded']
    oc_number = calculate_oc_number(row)
    return max(full_cars, oc_number)


def find_duplicate_numbers(series):
    return series.iloc[0]

terminal = 'Allouez'
train_data_as_dict = timetable(terminal)
print(train_data_as_dict)