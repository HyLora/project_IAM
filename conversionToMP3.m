% Imposta il percorso del file WAV di input e il nome del file MP3 di output
file_wav_input = '/babble.wav';
file_mp3_output = '/babble.mp3';

% Carica il file WAV
[y, Fs] = audioread(file_wav_input);

% Utilizza la funzione audiowrite per scrivere il file in formato MP3
audiowrite(file_mp3_output, y, Fs, 'Encoder', 'LAME');

disp('Conversione completata.');