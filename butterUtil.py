"""
butterUtil
Functions for multiple use.
2022 Knowblesse
"""
import numpy as np
from scipy.interpolate import interp1d

def vector2degree(r1,c1,r2,c2):
    """
    calculate the degree of the vector. 
    The vector is starting from (r1, c1) to (r2, c2). 
    Beware that the coordinate is not x, y rather it is row, column. 
    This row, column coordinate correspond to (inverted y, x). So, the 90 degree is the arrow going
    downward, and the 180 degree is the arrow going to the left.
    """
    # diagonal line
    l = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
    # temporal degree value
    temp_deg = np.rad2deg(np.arccos((c2 - c1) / l))
    # if r1 <= r2, then [0, 180) degree = temp_deg
    # if r1 > r2, then [180. 360) degree = 360 - temp_deg
    deg = 360 * np.array(r1 > r2, dtype=int) + (np.array(r1 <= r2, dtype=int) - np.array(r1 > r2, dtype=int)) * temp_deg
    return np.round(deg).astype(np.int)

def interpolateButterData(data):
    """
    Generate interpolated butter data.
    -----------------------------------------------------------------------------------------------
    data : ndarray(n x 3) : data of the labeled butter data
    -----------------------------------------------------------------------------------------------
    This function generate butter datapoints to all frames using interpolation method.
    However, first, the head direction value must be fixed.
        The head degree of an animal is stored in a range between 0 to 360 degree.
        If I use the raw head direction data from the labeled dataset,
        normal interpolation algorithm wouldn't correctly detect changes between near 0 or 360 degree.
        For example, if the head degree is changed from 350 to 10 degree, the interpolation
        algorithm would consider that the degree is decreased by 340, not increased by 20 degree.
        So it will output the mid-point value, 180, not 0.
        To compensate this, we need to add or subtract degree_offset_value to the original degree data.
        Since degree values used to draw lines does not be affected when I add or subtract 360 degree,
        these offset won't change the result.

    """
    prev_head_direction = data[0, 3]
    degree_offset_value = np.zeros(data.shape[0])
    for i in np.arange(1, data.shape[0]):
        # if the degree change is more than a half rotation, use the smaller rotation value instead.
        if np.abs(data[i, 3] - prev_head_direction) > 180:
            if data[i, 3] > prev_head_direction:
                degree_offset_value[i:] -= 360
            else:
                degree_offset_value[i:] += 360
        prev_head_direction = data[i, 3]

    # Generate Interpolated Label Data
    intp_x = interp1d(data[:, 0], data[:, 1], kind='linear')
    intp_y = interp1d(data[:, 0], data[:, 2], kind='linear')
    intp_d = interp1d(data[:, 0], np.convolve(data[:, 3] + degree_offset_value, np.ones(5), 'same') / 5, kind='linear')

    # Get num_frame
    num_frame = data[-1, 0]

    data_intp = np.stack([np.array(np.arange(num_frame)),
                          intp_x(np.array(np.arange(num_frame))),
                          intp_y(np.array(np.arange(num_frame))),
                          intp_d(np.array(np.arange(num_frame)))], axis=1)
    return data_intp
