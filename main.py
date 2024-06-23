from flask import Flask, render_template, request
import math

app = Flask(__name__)


# Functions for Question 1 calculations
def calculate_sampling_frequency(signal_bandwidth):
    return 2 * signal_bandwidth


def calculate_quantization_levels(number_of_quantizer_bits):
    return 2 ** number_of_quantizer_bits


def calculate_bit_rate_source_encoder(sampling_frequency, number_of_quantizer_bits, source_encoder_compression_rate):
    return sampling_frequency * number_of_quantizer_bits * source_encoder_compression_rate


def calculate_bit_rate_channel_encoder(bit_rate_source_encoder, channel_encoder_rate):
    return bit_rate_source_encoder / channel_encoder_rate


def calculate_bit_rate_interleaver(bit_rate_channel_encoder):
    return bit_rate_channel_encoder


# Functions for Question 2 calculations
def calculate_bits_per_resource_element(modulation_order):
    return int(modulation_order).bit_length() - 1


def calculate_bits_per_ofdm_symbol(bits_per_resource_element, num_subcarriers):
    return bits_per_resource_element * num_subcarriers


def calculate_bits_per_resource_block(bits_per_ofdm_symbol, num_ofdm_symbols):
    return bits_per_ofdm_symbol * num_ofdm_symbols


def calculate_max_transmission_rate(bits_per_resource_block, num_resource_blocks, duration_of_rb):
    return (bits_per_resource_block * num_resource_blocks) / duration_of_rb


# Route for the index page (Question 1)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Route for operations page
@app.route('/operations')
def operations():
    return render_template('operations.html')


# ***************************************************QUESTION1**********************************************************

# Route for Question 1 page
@app.route('/question1', methods=['GET', 'POST'])
def question1():
    if request.method == 'POST':
        try:
            signal_bandwidth = float(request.form['signal_bandwidth'])
            signal_bandwidth_unit = request.form['signal_bandwidth_unit']

            # Convert kHz to Hz if necessary
            if signal_bandwidth_unit == 'kHz':
                signal_bandwidth *= 1000

            number_of_quantizer_bits = int(request.form['number_of_quantizer_bits'])
            if number_of_quantizer_bits <= 0:
                raise ValueError("Number of quantizer bits must be a positive integer.")

            source_encoder_compression_rate = float(request.form['source_encoder_compression_rate'])
            channel_encoder_rate = float(request.form['channel_encoder_rate'])
            interleaver_number_of_bits = int(request.form['interleaver_number_of_bits'])
            if interleaver_number_of_bits <= 0:
                raise ValueError("Interleaver number of bits must be a positive integer.")

        except ValueError as e:
            return render_template('Question1.html', error=f"Invalid input: {e}")

        sampling_frequency = calculate_sampling_frequency(signal_bandwidth)
        quantization_levels = calculate_quantization_levels(number_of_quantizer_bits)
        bit_rate_source_encoder = calculate_bit_rate_source_encoder(sampling_frequency, number_of_quantizer_bits,
                                                                    source_encoder_compression_rate)
        bit_rate_channel_encoder = calculate_bit_rate_channel_encoder(bit_rate_source_encoder, channel_encoder_rate)
        bit_rate_interleaver = calculate_bit_rate_interleaver(bit_rate_channel_encoder)

        results = {
            "Sampling Frequency": f"{sampling_frequency} Hz",
            "Number of Quantization Levels": quantization_levels,
            "Bit Rate at the Output of the Source Encoder": f"{bit_rate_source_encoder} bits/sec",
            "Bit Rate at the Output of the Channel Encoder": f"{bit_rate_channel_encoder} bits/sec",
            "Bit Rate at the Output of the Interleaver": f"{bit_rate_interleaver} bits/sec"
        }

        return render_template('Question1.html', results=results)

    return render_template('Question1.html')


# ***************************************************QUESTION2**********************************************************

# Route for Question 2 page
@app.route('/question2', methods=['GET', 'POST'])
def question2():
    if request.method == 'POST':
        try:
            bandwidth_rb = float(request.form['bandwidth_rb'])
            if request.form['bandwidth_rb_unit'] == 'Hz':
                bandwidth_rb /= 1000  # Convert Hz to kHz

            subcarrier_spacing = float(request.form['subcarrier_spacing'])
            if request.form['subcarrier_spacing_unit'] == 'Hz':
                subcarrier_spacing /= 1000  # Convert Hz to kHz

            num_ofdm_symbols = int(request.form['num_ofdm_symbols'])

            duration_rb = float(request.form['duration_rb'])
            if request.form['duration_rb_unit'] == 'seconds':
                duration_rb *= 1000  # Convert seconds to milliseconds

            modulation_order = int(request.form['modulation_order'])
            num_parallel_rb = int(request.form['num_parallel_rb'])

            if not (modulation_order & (modulation_order - 1) == 0) and modulation_order != 0:
                raise ValueError("Modulation order must be a power of 2.")

        except ValueError as e:
            return render_template('Question2.html', error=f"Invalid input: {e}",
                                   bandwidth_rb=request.form['bandwidth_rb'],
                                   bandwidth_rb_unit=request.form['bandwidth_rb_unit'],
                                   subcarrier_spacing=request.form['subcarrier_spacing'],
                                   subcarrier_spacing_unit=request.form['subcarrier_spacing_unit'],
                                   num_ofdm_symbols=request.form['num_ofdm_symbols'],
                                   duration_rb=request.form['duration_rb'],
                                   duration_rb_unit=request.form['duration_rb_unit'],
                                   modulation_order=request.form['modulation_order'],
                                   num_parallel_rb=request.form['num_parallel_rb'])

        num_subcarriers = int(bandwidth_rb / subcarrier_spacing)
        bits_per_resource_element = calculate_bits_per_resource_element(modulation_order)
        bits_per_ofdm_symbol = calculate_bits_per_ofdm_symbol(bits_per_resource_element, num_subcarriers)
        bits_per_resource_block = calculate_bits_per_resource_block(bits_per_ofdm_symbol, num_ofdm_symbols)
        max_transmission_rate = calculate_max_transmission_rate(bits_per_resource_block, num_parallel_rb, duration_rb)

        results = {
            "Bits per Resource Element": f"{bits_per_resource_element}bits",
            "Bits per OFDM Symbol": f"{bits_per_ofdm_symbol} bits",
            "Bits per Resource Block": f"{bits_per_resource_block} bits",
            "Max Transmission Rate for User": f"{max_transmission_rate * 1000} bits/sec"
        }

        return render_template('Question2.html', results=results)

    return render_template('Question2.html')


# ***************************************************QUESTION3**********************************************************

EB_N0_MAP = {
    "BPSK/QPSK": {
        1e-1: 0,
        1e-2: 4,
        1e-3: 7,
        1e-4: 8.3,
        1e-5: 9.6,
        1e-6: 10.5,
        1e-7: 11.6,
        1e-8: 12,
    },
    "8-PSK": {
        1e-1: 0,
        1e-2: 6.5,
        1e-3: 10,
        1e-4: 12,
        1e-5: 12.5,
        1e-6: 14,
        1e-7: 14.7,
        1e-8: 15.6,
    },
    "16-PSK": {
        1e-1: 0,
        1e-2: 10.5,
        1e-3: 14.1,
        1e-4: 16,
        1e-5: 17.7,
        1e-6: 18.3,
        1e-7: 19.2,
        1e-8: 20,
    }
}


def convert_to_db(value, unit):
    if unit == "dB":
        return value
    elif unit == "watt":
        return 10 * math.log10(value)
    else:
        raise ValueError("Invalid unit")


def calculate_required_eb_n0(modulation_type, ber):
    if modulation_type in EB_N0_MAP:
        ber_map = EB_N0_MAP[modulation_type]
        for key in sorted(ber_map.keys(), reverse=True):
            if ber >= key:
                return ber_map[key]
    raise ValueError("Unsupported modulation type or BER")


def db_to_watt(db_value):
    """Convert decibel (dB) to watts."""
    return 10 ** (db_value / 10)


def calculate_total_transmit_power(L_p, G_t, G_r, A_t, A_r, N_f, T, L_f, L_o, F_margin, R, Eb_N0, M):
    K_dB = 10 * math.log10(1.38e-23)
    T_dB = 10 * math.log10(T)
    R_dB = 10 * math.log10(R)  # R is now always in bps

    P_t_dB = M - G_t - G_r - A_t - A_r + N_f + T_dB + K_dB + L_p + L_f + L_o + F_margin + R_dB + Eb_N0
    P_t_watt = db_to_watt(P_t_dB)

    return P_t_dB, P_t_watt


# Route for the index page (Question 3)
@app.route('/question3', methods=['GET', 'POST'])
def question3():
    if request.method == 'POST':
        try:
            # Retrieve form data including the selected units from radio buttons
            L_p = float(request.form['L_p'])
            L_p_unit = request.form['L_p_unit']
            G_t = float(request.form['G_t'])
            G_t_unit = request.form['G_t_unit']
            G_r = float(request.form['G_r'])
            G_r_unit = request.form['G_r_unit']
            R = float(request.form['R'])  # Data rate
            R_unit = request.form['R_unit']  # Unit of data rate

            # Convert R based on the selected unit
            if R_unit == 'kbps':
                R = R * 1e3  # Convert kbps to bps
            elif R_unit == 'Mbps':
                R = R * 1e6  # Convert Mbps to bps
            L_o = float(request.form['L_o'])
            L_o_unit = request.form['L_o_unit']
            L_f = float(request.form['L_f'])
            L_f_unit = request.form['L_f_unit']
            F_margin = float(request.form['F_margin'])
            F_margin_unit = request.form['F_margin_unit']
            A_t = float(request.form['A_t'])
            A_t_unit = request.form['A_t_unit']
            A_r = float(request.form['A_r'])
            A_r_unit = request.form['A_r_unit']
            N_f = float(request.form['N_f'])
            N_f_unit = request.form['N_f_unit']
            T = float(request.form['T'])  # Noise temperature is always in K
            M = float(request.form['M'])
            M_unit = request.form['M_unit']
            modulation_type = request.form['modulation_type']
            ber = float(request.form['ber'])

            # Convert values to dB based on selected units
            L_p_db = convert_to_db(L_p, L_p_unit)
            G_t_db = convert_to_db(G_t, G_t_unit)
            G_r_db = convert_to_db(G_r, G_r_unit)
            L_o_db = convert_to_db(L_o, L_o_unit)
            L_f_db = convert_to_db(L_f, L_f_unit)
            F_margin_db = convert_to_db(F_margin, F_margin_unit)
            A_t_db = convert_to_db(A_t, A_t_unit)
            A_r_db = convert_to_db(A_r, A_r_unit)
            N_f_db = convert_to_db(N_f, N_f_unit)
            M_db = convert_to_db(M, M_unit)

            Eb_N0 = calculate_required_eb_n0(modulation_type, ber)
            k = 228.6  # Boltzmann constant in dB (10 * log10(1.38e-23))
            P_t_dB, P_t_watt = calculate_total_transmit_power(L_p_db, G_t_db, G_r_db, A_t_db, A_r_db, N_f_db, T, L_f_db,
                                                              L_o_db, F_margin_db, R, Eb_N0, M_db)

            results = {
                "Total Transmit Power (P_t in dB)": f"{P_t_dB:.5f} dB",
                "Total Transmit Power (P_t in Watt)": f"{P_t_watt:.5f} Watts"
            }

            # Return results to template along with form data
            return render_template('Question3.html', results=results, form_data=request.form)

        except ValueError as e:
            return render_template('Question3.html', error=f"Invalid input: {e}", form_data=request.form)

    # Handle GET request by rendering the form initially
    return render_template('Question3.html', form_data={})


# ***************************************************QUESTION4**********************************************************
def calculate_throughput(mac_type, bandwidth_bps, prop_time_sec, frame_size_bits, frame_rate_fps):
    # Compute T (the time to transmit a single frame)
    T = frame_size_bits / bandwidth_bps

    # Compute G (offered load)
    G = frame_rate_fps * T

    # Compute alpha (normalized propagation delay)
    alpha = prop_time_sec / T

    # Handle different MAC types with respective formulas
    if mac_type == "unslotted_nonpersistent":
        numerator = G * math.exp(-2 * alpha * T)
        denominator = G * (1 + 2 * alpha) + math.exp(-alpha * G)
        throughput = numerator / denominator
    elif mac_type == "slotted_nonpersistent":
        numerator = alpha * G * math.exp(-2 * alpha * T)
        denominator = 1 - math.exp(-alpha * G) + alpha
        throughput = numerator / denominator
    elif mac_type == "unslotted_1_persistent":
        numerator = G * (1 + G + alpha * G * (1 + G + alpha * G / 2)) * math.exp(-G * (1 + 2 * alpha))
        denominator = G * (1 + 2 * alpha) - (1 - math.exp(-alpha * G)) + (1 + alpha * G) * math.exp(-G * (1 + alpha))
        throughput = numerator / denominator
    elif mac_type == "slotted_1_persistent":
        numerator = G * (1 + alpha - math.exp(-alpha * G)) * math.exp(-G * (1 + alpha))
        denominator = (1 + alpha) * (1 - math.exp(-alpha * G)) + alpha * math.exp(-G * (1 + alpha))
        throughput = numerator / denominator
    else:
        raise ValueError("Unsupported MAC type or functionality not implemented.")

    # Convert the throughput ratio to a percentage and prepare results
    throughput_percent = throughput * 100
    results = {
        "Throughput (%)": f"{throughput_percent:.3f}%",
        "Alpha": f"{alpha:.3f}",
        "T (sec)": f"{T:.6f}",
        "G (load)": f"{G:.3f}"
    }
    return results


@app.route('/question4', methods=['GET', 'POST'])
def question4():
    if request.method == 'POST':
        try:
            mac_type = request.form['mac_type']
            bandwidth = float(request.form['bandwidth'])
            bandwidth_unit = request.form['bandwidth_unit']
            prop_time = float(request.form['prop_time'])
            prop_time_unit = request.form['prop_time_unit']
            frame_size = float(request.form['frame_size'])
            frame_size_unit = request.form['frame_size_unit']
            frame_rate = float(request.form['frame_rate'])
            frame_rate_unit = request.form['frame_rate_unit']

            # Convert all inputs to base units
            if bandwidth_unit == 'Mbps':
                bandwidth *= 1e6
            elif bandwidth_unit == 'kbps':
                bandwidth *= 1e3

            if prop_time_unit == 'msec':
                prop_time *= 1e-3
            elif prop_time_unit == 'usec':
                prop_time *= 1e-6

            if frame_size_unit == 'kbit':
                frame_size *= 1e3

            if frame_rate_unit == 'kfps':
                frame_rate *= 1e3

            results = calculate_throughput(mac_type, bandwidth, prop_time, frame_size, frame_rate)
            return render_template('Question4.html', results=results, form_data=request.form)
        except ValueError as e:
            return render_template('Question4.html', error=f"Invalid input: {e}", form_data=request.form)
    return render_template('Question4.html', form_data={})


# ***************************************************QUESTION5**********************************************************
erlang_b_table = {
    0.1: [0.001, 0.046, 0.194, 0.439, 0.762, 1.1, 1.6, 2.1, 2.6, 3.1, 3.7, 4.2, 4.8, 5.4, 6.1, 6.7, 7.4, 8.0, 8.7, 9.4, 10.1, 10.8, 11.5, 12.2, 13.0, 13.7, 14.4, 15.2, 15.9, 16.7, 17.4, 18.2, 19.0, 19.7, 20.5, 21.3, 22.1, 22.9, 23.7, 24.4, 25.2, 26.0, 26.8, 27.6, 28.4],
    0.2: [0.002, 0.065, 0.249, 0.535, 0.900, 1.3, 1.8, 2.3, 2.9, 3.4, 4.0, 4.6, 5.3, 5.9, 6.6, 7.3, 7.9, 8.6, 9.4, 10.1, 10.8, 11.5, 12.3, 13.0, 13.8, 14.5, 15.3, 16.1, 16.8, 17.6, 18.4, 19.2, 20.0, 20.8, 21.6, 22.4, 23.2, 24.0, 24.8, 25.6, 26.4, 27.2, 28.1, 28.9, 29.7],
    0.5: [0.005, 0.105, 0.349, 0.701, 1.132, 1.6, 2.2, 2.7, 3.3, 4.0, 4.6, 5.3, 6.0, 6.7, 7.4, 8.1, 8.8, 9.6, 10.3, 11.1, 11.9, 12.6, 13.4, 14.2, 15.0, 15.8, 16.6, 17.4, 18.2, 19.0, 19.9, 20.7, 21.6, 22.4, 23.2, 24.0, 24.8, 25.7, 26.5, 27.4, 28.2, 29.1, 29.9, 30.8, 31.7],
    1.0: [0.010, 0.153, 0.455, 0.869, 1.361, 1.9, 2.5, 3.1, 3.8, 4.5, 5.2, 5.9, 6.6, 7.4, 8.1, 8.9, 9.7, 10.4, 11.2, 12.0, 12.8, 13.7, 14.5, 15.3, 16.1, 17.0, 17.8, 18.6, 19.5, 20.3, 21.2, 22.0, 22.9, 23.8, 24.6, 25.5, 26.4, 27.3, 28.1, 29.0, 29.9, 30.8, 31.7, 32.5, 33.4],
    1.2: [0.012, 0.168, 0.489, 0.922, 1.431, 2.0, 2.6, 3.2, 3.9, 4.6, 5.3, 6.1, 6.8, 7.6, 8.3, 9.1, 9.9, 10.7, 11.5, 12.3, 13.1, 14.0, 14.8, 15.6, 16.5, 17.3, 18.2, 19.0, 19.9, 20.7, 21.6, 22.5, 23.3, 24.2, 25.1, 26.0, 26.8, 27.7, 28.6, 29.5, 30.4, 31.3, 32.2, 33.1, 34.0],
    1.5: [0.020, 0.190, 0.530, 0.990, 1.520, 2.1, 2.7, 3.4, 4.1, 4.8, 5.5, 6.3, 7.0, 7.8, 8.6, 9.4, 10.2, 11.0, 11.8, 12.6, 13.5, 14.3, 15.2, 16.0, 16.9, 17.7, 18.6, 19.5, 20.3, 21.2, 22.1, 22.9, 23.8, 24.7, 25.6, 26.5, 27.4, 28.3, 29.2, 30.1, 31.0, 31.9, 32.8, 33.7, 34.6],
    2.0: [0.020, 0.223, 0.602, 1.092, 1.657, 2.3, 2.9, 3.6, 4.3, 5.1, 5.8, 6.6, 7.4, 8.2, 9.0, 9.8, 10.7, 11.5, 12.3, 13.2, 14.0, 14.9, 15.8, 16.6, 17.5, 18.4, 19.3, 20.2, 21.1, 22.0, 22.9, 23.8, 24.7, 25.6, 26.5, 27.4, 28.3, 29.2, 30.1, 31.0, 31.9, 32.8, 33.7, 34.6, 35.5],
    3.0: [0.031, 0.282, 0.715, 1.259, 1.875, 2.5, 3.2, 4.0, 4.7, 5.5, 6.3, 7.1, 8.0, 8.8, 9.6, 10.5, 11.4, 12.2, 13.1, 14.0, 14.9, 15.8, 16.7, 17.6, 18.5, 19.5, 20.4, 21.3, 22.2, 23.2, 24.1, 25.0, 26.0, 26.9, 27.9, 28.8, 29.8, 30.7, 31.7, 32.6, 33.6, 34.5, 35.5, 36.5, 37.4],
    5.0: [0.053, 0.381, 0.899, 1.525, 2.218, 3.0, 3.7, 4.5, 5.4, 6.2, 7.1, 8.0, 8.8, 9.7, 10.6, 11.5, 12.5, 13.4, 14.3, 15.2, 16.2, 17.1, 18.1, 19.0, 20.0, 20.9, 21.9, 22.9, 23.8, 24.8, 25.8, 26.7, 27.7, 28.7, 29.7, 30.7, 31.7, 32.7, 33.7, 34.7, 35.7, 36.7, 37.7, 38.7, 39.7]
}

def convert_to_watt(value, unit):
    if unit == "dB":
        return 10 ** (value / 10)
    elif unit == "watt":
        return value
    elif unit == "microwatt":
        return value * 1e-6
    else:
        raise ValueError("Invalid unit")


def convert_to_meters(value, unit):
    if unit == "km":
        return value * 1000
    elif unit == "m":
        return value
    else:
        raise ValueError("Invalid unit")


def convert_area_to_m2(value, unit):
    if unit == "km2":
        return value * 1e6
    elif unit == "m2":
        return value
    else:
        raise ValueError("Invalid unit")


def convert_calls_per_day_to_minutes(calls, unit):
    if unit == "day":
        return calls / (24 * 60)
    elif unit == "hour":
        return calls / 60
    elif unit == "minute":
        return calls
    elif unit == "second":
        return calls * 60
    else:
        raise ValueError("Invalid unit")


def convert_call_duration_to_minutes(duration, unit):
    if unit == "minute":
        return duration
    elif unit == "hour":
        return duration * 60
    elif unit == "second":
        return duration / 60
    elif unit == "day":
        return duration * 24 * 60
    else:
        raise ValueError("Invalid unit")


def calculate_max_distance(power_ref, distance_ref, path_loss_exp, receiver_sens):
    distance = distance_ref * (power_ref / receiver_sens) ** (1 / path_loss_exp)
    return distance


def calculate_cell_size(max_distance):
    return (3 * math.sqrt(3) * (max_distance ** 2)) / 2


def calculate_number_of_cells(area, cell_size):

    number_of_cells = area / cell_size
    return math.ceil(number_of_cells)


def calculate_traffic_load(subscribers, calls_per_minute, call_duration_minutes):
    total_calls = subscribers * calls_per_minute
    traffic_load = total_calls * call_duration_minutes
    return traffic_load


def calculate_traffic_load_per_cell(total_traffic_load, num_of_cells):
    return total_traffic_load / num_of_cells


def calculate_number_of_clusters(num_of_cells, path_loss_exp, sir):
    possible_n_values = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]
    threshold = (1 / 3) * ((6 * sir) ** (2 / path_loss_exp))

    for n in possible_n_values:
        if n >= threshold:
            return n

    raise ValueError("No suitable number of clusters found")


def calculate_min_carriers(gos, traffic_load_per_cell, timeslots_per_carrier):
    gos_column = erlang_b_table[gos]
    for channels, traffic in enumerate(gos_column, start=1):
        if traffic > traffic_load_per_cell:
            # Calculate required carriers
            carriers_needed = channels / timeslots_per_carrier
            # Return the ceiling of the number of carriers needed
            return math.ceil(carriers_needed)
    raise ValueError("No suitable number of carriers found")


@app.route('/question5', methods=['GET', 'POST'])
def question5():
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            area_unit = request.form['area_unit']
            area_m2 = convert_area_to_m2(area, area_unit)

            subscribers = int(request.form['subscribers'])
            calls_per_day = float(request.form['calls_per_day'])
            calls_per_day_unit = request.form['calls_per_day_unit']
            calls_per_minute = convert_calls_per_day_to_minutes(calls_per_day, calls_per_day_unit)

            call_duration = float(request.form['call_duration'])
            call_duration_unit = request.form['call_duration_unit']
            call_duration_minutes = convert_call_duration_to_minutes(call_duration, call_duration_unit)

            sir = float(request.form['sir'])
            sir_unit = request.form['sir_unit']
            sir_watt = convert_to_watt(sir, sir_unit)

            power_reference = float(request.form['power_reference'])
            power_reference_unit = request.form['power_reference_unit']
            power_reference_watt = convert_to_watt(power_reference, power_reference_unit)

            distance_reference = float(request.form['distance_reference'])
            distance_reference_unit = request.form['distance_reference_unit']
            distance_reference_m = convert_to_meters(distance_reference, distance_reference_unit)

            path_loss_exponent = float(request.form['path_loss_exponent'])
            receiver_sensitivity = float(request.form['receiver_sensitivity'])
            receiver_sensitivity_unit = request.form['receiver_sensitivity_unit']
            receiver_sensitivity_watt = convert_to_watt(receiver_sensitivity, receiver_sensitivity_unit)

            gos = float(request.form['gos'])

            timeslots_per_carrier = int(request.form['timeslots_per_carrier'])

            # Calculate maximum distance
            max_distance = calculate_max_distance(power_reference_watt, distance_reference_m, path_loss_exponent,
                                                  receiver_sensitivity_watt)

            # Calculate maximum cell size
            cell_size = calculate_cell_size(max_distance)

            # Calculate number of cells in the service area
            num_of_cells = calculate_number_of_cells(area_m2, cell_size)

            # Calculate traffic load in the whole cellular system in Erlangs
            total_traffic_load = calculate_traffic_load(subscribers, calls_per_minute, call_duration_minutes)

            # Calculate traffic load in each cell in Erlangs
            traffic_load_per_cell = calculate_traffic_load_per_cell(total_traffic_load, num_of_cells)

            # Calculate the number of cells in each cluster
            num_of_clusters = calculate_number_of_clusters(num_of_cells, path_loss_exponent,  sir_watt)

            # Calculate minimum number of carriers needed
            min_carriers = calculate_min_carriers(gos, traffic_load_per_cell, timeslots_per_carrier)
            total_min_carriers = min_carriers * num_of_clusters

            results = {
                "Maximum Distance (meters)": f"{max_distance:.3f}",
                "Maximum Cell Size (m²)": f"{cell_size:.2f}",
                "Number of Cells": f"{num_of_cells:.2f}",
                "Total Traffic Load (Erlangs)": f"{total_traffic_load:.2f}",
                "Traffic Load per Cell (Erlangs)": f"{traffic_load_per_cell:.2f}",
                "Number of Cells in Each Cluster": f"{num_of_clusters}",
                "Minimum Number of Carriers Needed": f"{total_min_carriers}"
            }

            return render_template('Question5.html', results=results)

        except ValueError as e:
            return render_template('Question5.html', error=f"Invalid input: {e}")

    return render_template('Question5.html')


if __name__ == "__main__":
    app.run(debug=True)
