import classes as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def remove_outliers(old_data):
    """Takes in Data object, removes all datapoints with residuals > 2 standard deviations out, returns cleaned data object"""
    df = old_data.table
    residuals = old_data.svd.residuals()
    df['residuals'] = residuals
    clean_df = df[abs(df.residuals) < 2 * residuals.std()].copy(deep=False)
    return cl.Data(clean_df, old_data.name)


def load_data(filename):
    """Reads data from a file, renames/adds new columns for 2019/2012, cleans data by removing outliers"""
    df = pd.read_csv(filename)
    if '2019' in filename:
        df.rename(columns={"elevation": "azimuth", "azimuth": "elevation"})
    if '2012' in filename:
        df['day'] = df['utc'].str[:10]
        df['month'] = df['utc'].str[5:7]
    original_data = cl.Data(df)
    clean_data = remove_outliers(original_data)
    return clean_data


def df_12month(df, month):
    """Get month of data from 2012. Enter month as e.g. '08'"""
    return df[df.month == month]


def df_12date(df, date):
    """Returns datapoints for a single night from 2012, enter date as 12-31 etc"""
    dayname = f'2012-{date}'
    return df[df.day == dayname]


def present_uncertainties(value, error):
    """Returns formatted string of value plus/minus error to the correct number of sig figs and in 
    scientific notation"""
    def find_exp(number) -> int:
        base10 = np.log10(abs(number))
        exponent = np.floor(base10)
        value = number/(10**exponent)
        if np.isnan(exponent):
            print(f"{number}")
            raise Exception("NaN error")
        return value, int(exponent)

    swapped = False
    if error > abs(value): # Ensures readable formatting if error > value
        swapped = True
        value, error = error, value

    value, value_exponent = find_exp(value)
    error, error_exponent = find_exp(error)
    exponent_difference = value_exponent - error_exponent

    if error > 2:
        value_roundto = exponent_difference
    else:
        value_roundto = exponent_difference + 1
    value = np.around(value, decimals=value_roundto)

    if error > 2:
        error = np.around(error, decimals=0)/10**exponent_difference
    else:
        error = np.around(error, decimals=1)/10**exponent_difference
    
    if not swapped:
        return f"{value} ± {error} E{value_exponent}"
    else:
        return f"{error} ± {value} E{value_exponent}"


def common_telpops(data_list):
    """Finds the POP settings common to all datasets in the given list of data"""
    common = set(data_list[0].telescopes.pops_reference)
    for data in data_list:
        telpops = set(data.telescopes.pops_reference)
        common = common.intersection(telpops)
    return common


def pops_difference(data_1, data_2, telpop_1, telpop_2):
    """Finds how much the difference between the two given POP delays changes between two nights"""
    diff_1 = data_1.svd.pops_values(telpop_1) - data_2.svd.pops_values(telpop_1)
    diff_2 = data_1.svd.pops_values(telpop_2) - data_2.svd.pops_values(telpop_2)
    return diff_1 - diff_2


def telpop_changes(data_list, refpop='W1-P2B3'):
    """W1-P2B3 is default reference POP, changes relative to first day in list given
    Finds changes in POP delays from night to night (only differences in delays are significant, hence need reference setting"""
    common = common_telpops(data_list)
    if refpop not in common:
        raise Exception("Reference POP not used in every night given")
    common.remove(refpop)
    for data in data_list[1:]:
        print(data.name + ":")
        for telpop in common:
            difference = pops_difference(data_list[0], data, refpop, telpop)
            print(f"{telpop} = {difference:.2e}")


def month_mean(data_1, data_2):
    """Returns mean and deviation uncertainty for two datasets"""
    positions_1 = data_1.svd.x()
    positions_2 = data_2.svd.x()
    mean_positions = 0.5 * (positions_1 + positions_2)
    errors = 0.5 * abs(positions_1 - positions_2)
    return mean_positions, errors


def vector_change(positions_1, positions_2, errors_1, errors_2):
    """Calculates distance moved and its uncertainty from two (3,1) vector positions and their errors
    Returns positions_1 - positions_2"""
    vector_diff = positions_1 - positions_2
    vector_sigma = np.sqrt(errors_1**2 + errors_2**2)
    return vector_diff, vector_sigma


def vector_changes_all(datalist_1, datalist_2, ref_data):
    """Calculates change+uncertainty between two pairs of datasets as vector and scalar distance, output Latex formatted"""
    mean1, err1 = month_mean(datalist_1[0], datalist_1[1])
    mean2, err2 = month_mean(datalist_2[0], datalist_2[1])
    telescopes = ref_data.telescopes.unique_telescopes[1:]
    for telescope in telescopes:
        i = telescopes.index(telescope)
        r, sigma_r = vector_change(mean1[i], mean2[i], err1[i], err2[i])
        rl, sigma_rl = length_change(mean1[i], mean2[i], err1[i], err2[i])
        x, y, z = r
        sx, sy, sz = sigma_r
        #print(f"{telescope}: {x:.1e} \pm {sx:.1e}, {y:.1e} \pm {sy:.1e}, {z:.1e} \pm {sz:.1e} \\\ ")
        print(f"{telescope} & ${present_uncertainties(x, sx)}$ & ${present_uncertainties(y, sy)}$ & ${present_uncertainties(z, sz)}$ & ${present_uncertainties(rl, sigma_rl)}$ \\\ ")


def length_change(positions_1, positions_2, errors_1, errors_2):
    """Calculates distance moved and its uncertainty from two vector positions for (3,1) array"""
    vector_diff, vector_sigma = vector_change(positions_1, positions_2, errors_1, errors_2)
    diff_r = np.sqrt((vector_diff**2).sum())
    sigma_r = 1/diff_r * np.sqrt(np.sum((vector_diff * vector_sigma)**2))
    return diff_r, sigma_r


def length_changes_all(datalist_1, datalist_2, ref_data):
    """Calculates change between two pairs of datasets as scalar distance, prints with uncertainties"""
    mean1, err1 = month_mean(datalist_1[0], datalist_1[1])
    mean2, err2 = month_mean(datalist_2[0], datalist_2[1])
    telescopes = ref_data.telescopes.unique_telescopes[1:]
    for telescope in telescopes:
        i = telescopes.index(telescope)
        r, sigma_r = length_change(mean1[i], mean2[i], err1[i], err2[i])
        print(f"{telescope}: {r:.1e} +/- {sigma_r:.1e} \\\ ")
