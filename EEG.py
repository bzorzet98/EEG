#%% Importamos las librerias
import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne as mne
from mne.preprocessing import (ICA,corrmap)
#%% Agregamos los archivos
files_path=["S01A.mat","S01B.mat","S02A.mat","S02B.mat","S03A.mat","S03B.mat","S04A.mat","S04B.mat",
        "S05A.mat","S05B.mat","S06A.mat","S06B.mat","S07A.mat","S07B.mat","S08A.mat","S08B.mat","S08C.mat",
        "S09A.mat","S09B.mat","S09C.mat","S10A.mat","S10B.mat","S10C.mat","S11A.mat","S11B.mat","S11C.mat",
        "S12A.mat","S12B.mat"]
nom=[s.replace(".mat", "") for s in files_path]
data=[]
for i in files_path:
        data.append(sio.loadmat(i))

#%% Acondicionamos los archivos
database={}
n_reg=len(data)
u2V=1e-6
i=0
for dat in data:
        aux={}
        aux['data_eeg']=dat['data']['X'][0][0]*u2V
        aux['frames']=aux['data_eeg'].shape[0]
        #Agregamos un canal de estimulación
        aux['data_eeg']=np.concatenate((aux['data_eeg'],np.zeros((aux['frames'],1))),axis=1)
        aux['labels']=dat['data']['y'][0][0]
        aux['trials']=dat['data']['trial'][0][0]
        aux['n_trials']=aux['trials'].shape[0]
        aux['fs']=dat['data']['fs'][0][0][0][0]
        database[nom[i]]=aux
        i+=1
channels=["FC3",'FCz','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CPz','CP4','STI 014']
n_channels=len(channels)
del i, aux
#%% EVENTOS
"""Creamos la matriz de eventos que necesitamos para poder definir las epocas.

Aca lo que defino son el inicio(0), la señal(+3), el segmento de imageneria motora(+3.5/+5.5) y la pausa()
"""
#Definimos el diccionario de eventos
events_dict=dict(hand=1,feet=2,cue=3,pausa=4,ini_imag_mot=5,fin_imag_mot=6)
#Definimos las muestras que hay entre etiqueta y etiqueta
t_ini_imag=0.5
t_fin_imag=2.5
delta_cue=database[nom[0]]['fs']*3
delta_p=database[nom[0]]['fs']*8
delta_ini=database[nom[0]]['fs']*(t_ini_imag)+delta_cue
delta_fin=database[nom[0]]['fs']*(t_fin_imag)+delta_cue
#Damos el formato para los eventos
for i in range(n_reg):
        n_trials=database[nom[i]]['n_trials']
        ev=np.concatenate((database[nom[i]]['trials'],
                        np.zeros([n_trials,1]),database[nom[i]]['labels']),axis=1)
        ev_cue=np.concatenate((database[nom[i]]['trials']+delta_cue,np.zeros([n_trials,1]),
                                np.ones([n_trials,1])*3),axis=1)
        ev_p=np.concatenate((database[nom[i]]['trials']+delta_p,np.zeros([n_trials,1]),
                        np.ones([n_trials,1])*4),axis=1)
        ev_p=ev_p[:-1]
        ev_ini=np.concatenate((database[nom[i]]['trials']+delta_ini,np.zeros([n_trials,1]),
                        np.ones([n_trials,1])*5),axis=1)
        ev_fin=np.concatenate((database[nom[i]]['trials']+delta_fin,np.zeros([n_trials,1]),
                        np.ones([n_trials,1])*6),axis=1)
        events=np.concatenate((ev,ev_cue,ev_p,ev_ini,ev_fin))
        events=np.array(sorted(events, key=lambda frame: frame[0]))
        database[nom[i]]['matrix_events']=events
        database[nom[i]]['ini_imag']=(database[nom[i]]['trials']+delta_ini)/database[nom[i]]['fs']
del i,events,n_trials,ev,ev_cue,ev_p,ev_fin,t_ini_imag,t_fin_imag,delta_cue,delta_p,delta_ini,delta_fin
#%% Damos el formato a los eventos
"""fig, axs = plt.subplots(13,figsize=[30,25])
t=np.linspace(events_frames[0,0]*ts_eeg,events_frames[1,0]*ts_eeg,-events_frames[0,0]+events_frames[1,0])
for i in range(13):
        axs[i].plot(t,data_eeg[events_frames[0,0]:events_frames[1,0],i])
        #axs[i].set_title(channels[i])
        axs[i].set_xlabel("Tiempo [s]")
        axs[i].set_ylabel(channels[i])
        axs[i].vlines(x=events_frames[0,0]*ts_eeg,ymin=-50,ymax=50, color='black')
        axs[i].vlines(x=events_frames[0,1]*ts_eeg,ymin=-50,ymax=50, color='red')
        axs[i].vlines(x=events_frames[0,2]*ts_eeg, ymin=-50,ymax=50,color='green')
plt.show()"""
#%% Creamos la estructura INFO
info= mne.create_info(
        ch_names=channels,
        ch_types=['eeg']*(n_channels-1)+['stim'],
        sfreq=database[nom[0]]['fs'])
info.set_montage('standard_1020')
#%% CREAMOS LOS RAW
data_raw={}
for i in range(n_reg):
        data_raw[nom[i]]=mne.io.RawArray(database[nom[i]]['data_eeg'].T,info)
        #print(database[nom[i]]['matrix_events'][-1,:])
        data_raw[nom[i]].add_events(database[nom[i]]['matrix_events'],stim_channel='STI 014')
#%% Graficamos 1 RAW
ev_color={1: 'r', 2: 'g', 3: 'b', 4: 'm',5:'b',6:'b'}
"""scaling_eeg=20e-6
i=0

data_raw[nom[i]].plot(events=database[nom[i]]['matrix_events'], scalings=scaling_eeg,\
        color='gray',event_color=ev_color,start=database[nom[i]]['matrix_events'][0,0]/database[nom[i]]['fs'],\
        duration=(database[nom[i]]['matrix_events'][5,0]-database[nom[i]]['matrix_events'][0,0])/database[nom[i]]['fs'])"""
# %% Creamos anotaciones
for i in range(n_reg):
        inicio=database[nom[i]]['ini_imag'].flatten()
        anot=mne.Annotations(onset=inicio, # in seconds
                duration=2, # in seconds, too
                description="IM")
        data_raw[nom[i]].set_annotations(anot)
# %% PLOTEAMOS LOS RAWS
"""scaling_eeg=20e-6
i=0
data_raw[nom[i]].plot(scalings=scaling_eeg,color='gray',\
        start=database[nom[i]]['matrix_events'][0,0]/database[nom[i]]['fs'],\
                duration=2+(database[nom[i]]['matrix_events'][10,0]-database[nom[i]]['matrix_events'][0,0])/database[nom[i]]['fs'])"""
# %% Ploteamos la ubicacion de los sensores
"""i=0
data_raw[nom[i]].plot_sensors(ch_type='eeg', show_names=channels[:-1])
"""
#%% Creamos las eocas
"""Para crear las epocas, lo unico que hay que hacer es darle los eventos que queremos que tome en cuenta.

En este caso , voy a decirle que inicie en el inicio de la imagineria y que finalice pasando los 2 segundos. """
data_epochs={}
event_imag={'ini_imag':5}
t_ini=-0.2
t_end=2
for i in nom:
        data_epochs[i]=mne.Epochs(data_raw[i], events=database[i]['matrix_events'].astype(int),
                event_id=event_imag,  tmin=t_ini, tmax=t_end )
#%% Visualizamos las epocas
"""i=0
data_epochs[nom[i]].plot(picks='eeg',scalings=20e-6,n_epochs=2,events=database[nom[i]]['matrix_events'],
                        event_color=ev_color,
                        group_by='position', butterfly=True)"""
#%% Ploteamos la ubicacion de los sensores
"""i=0
data_epochs[nom[i]].plot_sensors(kind='3d', ch_type='eeg', show_names=channels[:-1])"""
#%% Graficamos la densidad espectral de las epocas
"""i=0
data_epochs[nom[i]].plot_psd(picks='eeg',tmax=np.inf, fmax=database[nom[i]]['fs']/2)"""

#%% VIsualizamos la densidad espectral
"""i=0
data_epochs[nom[i]].plot_image(picks='eeg', combine='mean')
"""
#%% Realizamos algunos Procesamientos a las señales
"""SIguiendo los tutoriales de mne, lo que hacemos ver la potencia de la 
señal para identificar ruido de linea principalmente. Supuestamente, 
la base de datos fue filtrada a 50Hz con un filtro Notch. """
#%% Grafica del espectro de potencias de las señales para el registro i de todos los canales.

"""i=0
fig = data_raw[nom[i]].plot_psd(tmax=np.inf, fmax=database[nom[i]]['fs']/2, average=False)
print(fig.axes)"""
"""ax=fig.axes[0]
freqs = ax.lines[-1].get_xdata()
psds = ax.lines[-1].get_ydata()
for freq in (50, 100, 150, 200,250):
        ax.hvline(x=freq, ymin=psds.min, ymax=psds.max, color='red',
        linewidth=1)"""
#%% PLoteamos la media de todos los registros
"""for i in range(n_reg):
        fig = data_raw[nom[i]].plot_psd(tmax=np.inf, fmax=database[nom[i]]['fs']/2, average=True)"""
#%% FIltramos los ruidos de linea
"""Para esto usamos la funcion notch_filter(), 
Aca si no seteamos la ventana por defecto es de fm/200 (). 
Hacemos un filto de fase cero, para no generar 
cambios en la señal (se aplica el filtrado en los dos sentidos)"""
freqs = (50, 100, 150, 200, 250)
data_raw_notch={}
for i in nom:
        data_raw_notch[i]= data_raw[i].copy().notch_filter(freqs=freqs, picks='eeg',phase='zero-double')
        """for title, data in zip(['Un', 'Notch '], [data_raw[i], data_raw_notch[i]]):
                fig = data.plot_psd(fmax=250, average=True)
                fig.subplots_adjust(top=0.85)
                fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
                fig.show()"""
#%% FIltramos los Raws, elimanando las bajas frecuencias
data_raw_filt={}
for i in nom:
        data_raw_filt[i] = data_raw[i].copy().filter(l_freq=1, h_freq=None)
#%% Ploteamos la densidad espectral de los datos fitrados
"""for i in range(n_reg):
        fig = data_raw_filt[nom[i]].plot_psd(tmax=np.inf, fmax=database[nom[i]]['fs']/2, average=True)
"""
#%% ICA
icas={}
for i in nom:
        icas[i]=ICA(n_components=13, max_iter='auto',random_state=51)
        icas[i].fit(data_raw_filt[i])
#%% VISUALIZACION
for i in nom:
        icas[i].plot_sources(data_raw[i], show_scrollbars=False,)
        
#%%  PLoteamos las componentes 
for i in nom:
        icas[i].plot_components()

a=10


# %%Eliminamos las componenetes del ica
"""ica.exclude = [0, 1]  # indices chosen based on various plots above
reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
         show_scrollbars=False)
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
                 show_scrollbars=False)"""