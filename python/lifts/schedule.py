import pandas as pd

def find_duplicate_numbers(series):
    duplicates = series[series.duplicated(keep=False)]
    if not duplicates.empty:
        return ''.join(map(str, duplicates.unique()))
    else:
        return series.iloc[0]


def train_timetable(terminal):
    df = pd.read_csv('C:/Users/Irena Tong/PycharmProjects/simulation_test/data/train_consist_plan.csv')
    df_terminal = df[(df['Destination_ID'] == terminal) & (df['Train_Type'] == 'Intermodal')]  # Filter by the selected terminal and intermodal type railcar

    df_grouped = df_terminal.groupby('Train_ID').agg({
        'Cars_Loaded': find_duplicate_numbers,
        'Cars_Empty': find_duplicate_numbers,
        'Arrival_Time_Actual_Hr': 'min',
        'Departure_Time_Actual_Hr': 'min'
    }).reset_index()

    df_grouped_sorted = df_grouped.sort_values(by='Arrival_Time_Actual_Hr', ascending=True)

    return df_grouped_sorted


def next_train_timetable(train_id, terminal):
    df_terminal = train_timetable(terminal)
    df_next_train = df_terminal.iloc[train_id]
    return df_next_train


def outbound_containers():
    df = pd.read_csv('C:/Users/Irena Tong/PycharmProjects/simulation_test/data/outbound_plan.csv')
    return df


def get_next_train_outbound_data(index):
    outbound_df = outbound_containers()
    outbound_num = outbound_df.iloc[index]['Outbound_Num']
    return outbound_num

# # Test codes
# terminal = 'Allouez'
# print(train_timetable(terminal))
#
# next_train = next_train_timetable(1, terminal)
# print(next_train)
#
# next_outbound_num = get_next_train_outbound_data(1)
# print(next_outbound_num)