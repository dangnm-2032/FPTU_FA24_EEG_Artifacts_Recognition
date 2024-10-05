% Step 1: Load the CSV file
data = readtable('left_1.csv'); % Replace 'yourfile.csv' with your CSV file name

% Step 2: Verify data is loaded correctly
size(data)  % Check the dimensions of the loaded data

% Step 3: Extract time-domain signal (ensure you reference the correct column)
signal = data{:, "AF7"};  % Assuming the signal is in the second column (change if needed)

% Step 4: Check if signal is non-empty
if isempty(signal)
    error('The signal is empty or not loaded correctly. Check your data file.');
end

% Step 5: Define sampling frequency (fs) - adjust as necessary
fs = 214;  % Replace with the actual sampling frequency of your signal

% Step 6: Apply FFT to convert to frequency domain
L = length(signal);  % Length of the signal
Y = fft(signal);     % Compute the FFT

% Step 7: Compute the two-sided spectrum and single-sided spectrum
P2 = abs(Y/L);  % Two-sided spectrum
P1 = P2(1:L/2+1);  % Single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1);

% Step 8: Define frequency domain f
f = fs*(0:(L/2))/L;

% Step 9: Plot the frequency spectrum
figure;
plot(f, P1)
title('Single-Sided Amplitude Spectrum of Signal')
xlabel('Frequency (Hz)')
ylabel('|P1(f)|')
grid on;
