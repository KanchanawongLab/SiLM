import sys, numpy as np, pandas as pd
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import lib.SGI as lib, refractiveIndices.refractiveIndices as refractiveIndices
    
MATPLOTLIB_CTABLE=['nipy_spectral', 'nipy_spectral_r','gist_ncar','gist_rainbow', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap','hot', 'afmhot',
                   'rainbow','rainbow_r','jet','jet_r','viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
             'cubehelix', 'brg' ]
TEXT_COLOR = ['red','yellow','green','blue','cyan','black','white']
PLOT_OPTION = ['Main plot only', 'Extra']

DEFAULT_PARAMETERS = {"Excitation Wavelength": 642.0, "SiO2 Thickness": 500.0, "SiO2 refractive index":1.46310,
                      "Si refractive index": 4.367, "Buffer refractive index": 1.3404,"# Angle-points":15,"Expt. Angle Int.":4,"Min. Expt. Angle":0,
                      "Maximum Incidence Angle":52.0, "Incidence Angle Interval":1.0, "Maximum Z":250, "Z Interval": 1.0, "Display Z Maximum:":150.0,
                      "Angle Offset":0, "Guess Iteration": 10, "Guess Maximum":160,"Pixel Size": 106.667, "Camera Offset":100,
                      "Expt. Angle Offset":0. , "GPU Iteration": 50, "Max GPU Fits": 5e6, "Glass Index":1.523, "Wavelength Interval":2,
                      "Maximum Wavelength":700, "Minimum Wavelength":450}

MAINKEY_STYLE = 'QPushButton {background-color: #FFFFAA; color: red;}'
MAINKEY_STYLE_2 = 'QPushButton {background-color: #FF7777; color: black;}'
PARAM_STYLE_A = 'QDoubleSpinBox {background-color : #DDDDFF;}'
PARAM_STYLE_B = 'QDoubleSpinBox {background-color : #DDFFDD;}'
DROPLIST_STYLE_A = 'QComboBox {background-color : #DDFFDD;}'

def get_positive(param):
    offset,amplitude, period, phase = param
    if amplitude <0:
        phase = phase+np.pi
    if period <0:
        period = np.abs(period)
    return [offset,amplitude,period,phase]

def save_dataframe_csv(window,*, filename = None, dataframe = None, datadict = None, mode = None,  prompt = "Save drift to *DRIFT.csv file", filter = "*DRIFT.csv" ):
    if filename is None:
        if hasattr(window,'datafile'):
            if window.datafile is None:
                fn = 'null'
            else:
                fn = window.datafile
        else:
             fn = 'null'   
        h5name, h5path = lib.getPALM_extName(fn, mode=mode)
        filename = QtWidgets.QFileDialog.getSaveFileName(window, prompt, h5path, filter=filter)   
        filename = filename[0]
    if filename is None :
        return
    
    if dataframe is not None:
        dataframe.to_csv(filename, index=  False)
        print('Saving.. .csv file:', filename)
        return filename
    if datadict is not None:
        dictpd = pd.DataFrame(datadict)
        dictpd.to_csv(filename, index=  False)
        print('Saving.. .csv file:', filename)
        return filename
    
class SGI(QtWidgets.QMainWindow):

    windowSize = [1660,1080-40]
    viewportSize = [1024-24,1024-24]
    viewportPosition = [10+300,30]
        
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SiLM Field 0.99(c)2021-5 Nanoscale Mechanobiology Laboratory, Mechanobiology Institute, National University of Singapore")

        self.resize(self.windowSize[0], self.windowSize[1])
        pwd = []
        self.pwd = pwd
        
        self.reset_canvas() 
                  
        menu_bar = self.menuBar()
        utilities_menu = menu_bar.addMenu("Utilities")
        export_calib_action = utilities_menu.addAction("Export calibration curves (.csv)")
        export_calib_action.triggered.connect(self.exportCalib)
        closeall_action =  utilities_menu.addAction("Close all auxiliary plots")
        closeall_action.triggered.connect(lambda x: plt.close('all'))
        
        self.pcolorDroplist = QtWidgets.QComboBox(self)
        self.pcolorDroplist.setGeometry(150+5-130,5+35,125,30)
        self.pcolorDroplist.addItems(MATPLOTLIB_CTABLE)
        self.pcolorDroplist.setCurrentIndex(MATPLOTLIB_CTABLE.index('gnuplot'))
        self.pcolorDroplist.currentIndexChanged.connect(self.chooseLUT)
        self.plotOptionDroplist = QtWidgets.QComboBox(self)
        self.plotOptionDroplist.setGeometry(150+5,5+35,125,30)
        self.plotOptionDroplist.addItems(PLOT_OPTION)
##        Tab Definition
        
        self.leftTab = QtWidgets.QTabWidget(self)
        self.leftTab3 = QtWidgets.QWidget(self.leftTab)
        # self.leftTab2 = QtWidgets.QWidget(self.leftTab)
        self.leftTab.addTab(self.leftTab3,"Simulation ")
        # self.leftTab.addTab(self.leftTab2,"CRLB-z ")
        self.leftTab.setGeometry(10,30+50,270,self.viewportSize[1]-60) 
        
    #### Simulation tab 12-4-2020
        self.simulationLabel = QtWidgets.QLabel("Excitation Field Simulation Parameters:", self.leftTab3)
        self.simulationLabel.setGeometry(5, 5, 200, 30)
        self.maxAngleLabel = QtWidgets.QLabel("Max Angle(deg):", self.leftTab3)
        self.maxAngleLabel.setGeometry(5, 5+35, 80, 30)
        self.maxAngleSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.maxAngleSlider.setGeometry(85, 5+35, 100, 30);        self.maxAngleSlider.setRange(0,70)
        self.maxAngleSlider.setValue(int(DEFAULT_PARAMETERS["Maximum Incidence Angle"]))
        self.maxAngleSlider.setSingleStep(1)
        self.maxAngleSlider.setPageStep(20)        
        self.maxAngleSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.maxAngleSpinBox.setGeometry(190,5+35,70,30)
        self.maxAngleSpinBox.setMinimum(0);        self.maxAngleSpinBox.setMaximum(70)
        self.maxAngleSpinBox.setSingleStep(1);        self.maxAngleSpinBox.setValue(DEFAULT_PARAMETERS["Maximum Incidence Angle"])
        self.maxAngleSlider.valueChanged.connect(self.maxangle_slider)
        self.maxAngleSpinBox.valueChanged.connect(self.maxangle_spinbox)

        self.intervalLabel = QtWidgets.QLabel("Interval(deg):", self.leftTab3)
        self.intervalLabel.setGeometry(5, 5+35*2, 80, 30)
        self.intervalSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.intervalSlider.setGeometry(85, 5+35*2, 100, 30)
        self.intervalSlider.setRange(1,15)
        self.intervalSlider.setValue(int(DEFAULT_PARAMETERS["Incidence Angle Interval"]))
        self.intervalSlider.setSingleStep(1)
        self.intervalSlider.setPageStep(20)        
        self.intervalSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.intervalSpinBox.setGeometry(190,5+35*2,70,30)
        self.intervalSpinBox.setMinimum(0.1)
        self.intervalSpinBox.setMaximum(15.)
        self.intervalSpinBox.setSingleStep(1)
        self.intervalSpinBox.setValue(DEFAULT_PARAMETERS["Incidence Angle Interval"])
        self.intervalSlider.valueChanged.connect(self.interval_slider)
        self.intervalSpinBox.valueChanged.connect(self.interval_spinbox)

        self.maxZLabel = QtWidgets.QLabel("Z-ceiling (nm):", self.leftTab3)
        self.maxZLabel.setGeometry(5, 5+35*3, 80, 30)
        self.maxZSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.maxZSlider.setGeometry(85, 5+35*3, 100, 30)
        self.maxZSlider.setRange(100,3000)
        self.maxZSlider.setValue(int(DEFAULT_PARAMETERS["Maximum Z"]))
        self.maxZSlider.setSingleStep(1);        self.maxZSlider.setPageStep(20)        
        self.maxZSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.maxZSpinBox.setGeometry(190,5+35*3,70,30)
        self.maxZSpinBox.setMinimum(100);        self.maxZSpinBox.setMaximum(3000)
        self.maxZSpinBox.setSingleStep(1)
        self.maxZSpinBox.setValue(DEFAULT_PARAMETERS["Maximum Z"])
        self.maxZSlider.valueChanged.connect(self.maxz_slider)
        self.maxZSpinBox.valueChanged.connect(self.maxz_spinbox)

        self.zIntervalLabel = QtWidgets.QLabel("Z-interval (nm):", self.leftTab3)
        self.zIntervalLabel.setGeometry(5, 5+35*4, 80, 30)
        self.zIntervalSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.zIntervalSlider.setGeometry(85, 5+35*4, 100, 30)
        self.zIntervalSlider.setRange(1,10)
        self.zIntervalSlider.setValue(int(DEFAULT_PARAMETERS["Z Interval"]))
        self.zIntervalSlider.setSingleStep(1);        self.zIntervalSlider.setPageStep(1)        
        self.zIntervalSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.zIntervalSpinBox.setGeometry(190,5+35*4,70,30)
        self.zIntervalSpinBox.setMinimum(0.1);        self.zIntervalSpinBox.setMaximum(10);        self.zIntervalSpinBox.setSingleStep(1)
        self.zIntervalSpinBox.setValue(DEFAULT_PARAMETERS["Z Interval"])
        self.zIntervalSlider.valueChanged.connect(self.zinterval_slider)
        self.zIntervalSpinBox.valueChanged.connect(self.zinterval_spinbox)

        self.fieldSimulateButton = QtWidgets.QPushButton("Simulate Field Intensity(Z, Angle)",self.leftTab3, clicked = self.fieldSimulate2D )
        self.fieldSimulateButton.setGeometry(5,5+35*5,250,30)

        self.specificZLabel = QtWidgets.QLabel("Z (nm):", self.leftTab3)
        self.specificZLabel.setGeometry(5, 15+35*7, 80, 30)
        self.specificZSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.specificZSlider.setGeometry(85, 15+35*7, 100, 30)
        self.specificZSlider.setRange(1,5000);        self.specificZSlider.setValue(100)
        self.specificZSlider.setSingleStep(1);        self.specificZSlider.setPageStep(1)        
        self.specificZSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.specificZSpinBox.setGeometry(190,15+35*7,70,30)
        self.specificZSpinBox.setMinimum(0.1);        self.specificZSpinBox.setMaximum(5000)
        self.specificZSpinBox.setSingleStep(1);        self.specificZSpinBox.setValue(100)
        self.specificZSlider.valueChanged.connect(self.specificz_slider)
        self.specificZSpinBox.valueChanged.connect(self.specificz_spinbox)
        self.fieldSimulate1DButton = QtWidgets.QPushButton("Plot Field Intensity@Z vs Angle",self.leftTab3, clicked = self.fieldSimulate1D )
        self.fieldSimulate1DButton.setGeometry(5,15+35*8,250,30)
        
        self.specificALabel = QtWidgets.QLabel("Angle (deg):", self.leftTab3)
        self.specificALabel.setGeometry(5, 15+35*10, 80, 30)
        self.specificASlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.specificASlider.setGeometry(85, 15+35*10, 100, 30)
        self.specificASlider.setRange(0,60);        self.specificASlider.setValue(0)
        self.specificASlider.setSingleStep(1);        self.specificASlider.setPageStep(1)        
        self.specificASpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.specificASpinBox.setGeometry(190,15+35*10,70,30)
        self.specificASpinBox.setMinimum(0.0);        self.specificASpinBox.setMaximum(60)
        self.specificASpinBox.setSingleStep(1);        self.specificASpinBox.setValue(0)
        self.specificASlider.valueChanged.connect(self.specificA_slider)
        self.specificASpinBox.valueChanged.connect(self.specificA_spinbox)
        self.fieldSimulate1AButton = QtWidgets.QPushButton("Plot Field Intensity@Angle vs Z",self.leftTab3, clicked = self.fieldSimulate1A )
        self.fieldSimulate1AButton.setGeometry(5,15+35*11, 250,30)
        self.fieldSimulate1DStandingWaveButton = QtWidgets.QPushButton("Plot Field Intensity@Z vs Angle (Wave)",self.leftTab3, clicked= self.waveSimulate2D)
        self.fieldSimulate1DStandingWaveButton.setGeometry(5,15+35*12,250,30)
        self.fluorophoreSimulateMultiButton = QtWidgets.QPushButton("Simulate Fluorophores@Multi-Z vs Angle",self.leftTab3, clicked = self.simulateFluorophoreZ)
        self.fluorophoreSimulateMultiButton.setGeometry(5,15+35*13, 250,30)
        
        self.multiALabel = QtWidgets.QLabel("Multi-Angle (comma delimited):", self.leftTab3)
        self.multiALabel.setGeometry(5, 15+35*15, 150, 30)
        self.multiAngleLineEdit = QtWidgets.QLineEdit(self.leftTab3)
        self.multiAngleLineEdit.setGeometry(160,15+35*15,100,30)
        self.multiAngleLineEdit.setText('12,34,48')
        self.fieldSimulateMultiButton = QtWidgets.QPushButton("Plot Field Intensity@Multi-Angle vs Z",self.leftTab3, clicked = self.fieldSimulateMulti)
        self.fieldSimulateMultiButton.setGeometry(5,15+35*16, 250,30)
        
        self.maxLambdaLabel = QtWidgets.QLabel("Max Wavelength(nm):", self.leftTab3)
        self.maxLambdaLabel.setGeometry(5, 15+35*18, 110, 30)
        self.maxLambdaSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.maxLambdaSlider.setGeometry(115, 15+35*18, 70, 30);        self.maxLambdaSlider.setRange(300,1000)
        self.maxLambdaSlider.setValue(int(DEFAULT_PARAMETERS["Maximum Wavelength"]))
        self.maxLambdaSlider.setSingleStep(1)
        self.maxLambdaSlider.setPageStep(20)        
        self.maxLambdaSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.maxLambdaSpinBox.setGeometry(190,15+35*18,70,30)
        self.maxLambdaSpinBox.setMinimum(300);        self.maxLambdaSpinBox.setMaximum(1000)
        self.maxLambdaSpinBox.setSingleStep(1);        self.maxLambdaSpinBox.setValue(DEFAULT_PARAMETERS["Maximum Wavelength"])
        self.maxLambdaSlider.valueChanged.connect(self.maxlambda_slider)
        self.maxLambdaSpinBox.valueChanged.connect(self.maxlambda_spinbox)
        
        self.minLambdaLabel = QtWidgets.QLabel("Min. Wavelength(nm):", self.leftTab3)
        self.minLambdaLabel.setGeometry(5, 15+35*20, 110, 30)
        self.minLambdaSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.minLambdaSlider.setGeometry(115, 15+35*20, 70, 30);        self.minLambdaSlider.setRange(300,1000)
        self.minLambdaSlider.setValue(int(DEFAULT_PARAMETERS["Maximum Wavelength"]))
        self.minLambdaSlider.setSingleStep(1)
        self.minLambdaSlider.setPageStep(20)        
        self.minLambdaSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.minLambdaSpinBox.setGeometry(190,15+35*20,70,30)
        self.minLambdaSpinBox.setMinimum(300);        self.minLambdaSpinBox.setMaximum(1000)
        self.minLambdaSpinBox.setSingleStep(1);        self.minLambdaSpinBox.setValue(DEFAULT_PARAMETERS["Minimum Wavelength"])
        self.minLambdaSlider.valueChanged.connect(self.minlambda_slider)
        self.minLambdaSpinBox.valueChanged.connect(self.minlambda_spinbox)
        
        self.l_intervalLabel = QtWidgets.QLabel("Wvlgth Int.(nm):", self.leftTab3)
        self.l_intervalLabel.setGeometry(5, 15+35*19, 110, 30)
        self.l_intervalSlider = QtWidgets.QSlider(Qt.Horizontal, self.leftTab3)
        self.l_intervalSlider.setGeometry(115, 15+35*19, 70, 30)
        self.l_intervalSlider.setRange(1,15)
        self.l_intervalSlider.setValue(int(DEFAULT_PARAMETERS["Wavelength Interval"]))
        self.l_intervalSlider.setSingleStep(1)
        self.l_intervalSlider.setPageStep(20)        
        self.l_intervalSpinBox = QtWidgets.QDoubleSpinBox(self.leftTab3)
        self.l_intervalSpinBox.setGeometry(190,15+35*19,70,30)
        self.l_intervalSpinBox.setMinimum(0.1)
        self.l_intervalSpinBox.setMaximum(15.)
        self.l_intervalSpinBox.setSingleStep(1)
        self.l_intervalSpinBox.setValue(DEFAULT_PARAMETERS["Wavelength Interval"])
        self.l_intervalSlider.valueChanged.connect(self.l_interval_slider)
        self.l_intervalSpinBox.valueChanged.connect(self.l_interval_spinbox)
        self.simulate3DButton = QtWidgets.QPushButton("Simulate Field vs Z,A,Lambda",self.leftTab3, clicked = self.fieldSimulate3D)
        self.simulate3DButton.setGeometry(5,15+35*21, 250,30)
        
        self.clearScreenButton = QtWidgets.QPushButton("Clear Screen",self.leftTab3, clicked = self.clearScreen)
        self.clearScreenButton.setGeometry(5,15+35*24, 250,30)
        
##Right-tab
        self.rightTab = QtWidgets.QTabWidget(self)
        self.rightTab1 = QtWidgets.QWidget(self.rightTab)
        self.rightTab2 = QtWidgets.QWidget(self.rightTab)
        self.rightTab.addTab(self.rightTab1,"Experimental Parameters ")
        self.rightTab.addTab(self.rightTab2,"CRLB-z ")
        self.rightTab.setGeometry(self.viewportSize[0]+self.viewportPosition[0]+20,80,320,self.viewportSize[1]-210)
        
        self.CRLBzbutton = QtWidgets.QPushButton("Plot CRLB-z",self.rightTab2, clicked = self.plotCRLBz)
        self.CRLBzbutton.setGeometry(5,10, 150,30)
        self.CRLBzSearchbutton = QtWidgets.QPushButton("Angles +/- 3 ",self.rightTab2, clicked = self.plotCRLBz_variation)
        self.CRLBzSearchbutton.setGeometry(5+155,10, 150,30)
        
        self.IsigLabel = QtWidgets.QLabel("I-sig (Photons):", self.rightTab2)
        self.IsigLabel.setGeometry(5, 10+40, 120, 30)
        
        self.IsigLineEdit = QtWidgets.QLineEdit(self.rightTab2)
        self.IsigLineEdit.setGeometry(5+155,10+40,120,30)
        self.IsigLineEdit.setText('100')
        
        self.bLabel = QtWidgets.QLabel("background (Photons):", self.rightTab2)
        self.bLabel.setGeometry(5, 10+40+35, 120, 30)
        
        self.bLineEdit = QtWidgets.QLineEdit(self.rightTab2)
        self.bLineEdit.setGeometry(5+155,10+40+35,120,30)
        self.bLineEdit.setText('10')
        
        self.add_ExperimentalParameters(self.rightTab1)
        self.init_parameters()

        self.autoRIButton = QtWidgets.QPushButton("Auto-Lookup Refractive Indices",self.rightTab1)
        self.autoRIButton.setGeometry(5,880-150,300,30)
        self.autoRIButton.setStyleSheet(MAINKEY_STYLE)
        self.autoRIButton.clicked.connect(self.autoRI)
        
        self.show()
    
    def add_ExperimentalParameters(self, thistab):
        self.excitationWavelengthLabel = QtWidgets.QLabel("Excite.(nm):", thistab)
        self.excitationWavelengthLabel.setGeometry(5, 10, 60, 30)
        self.excitationWavelengthSlider = QtWidgets.QSlider(Qt.Horizontal, thistab)
        self.excitationWavelengthSlider.setGeometry(65, 10, 160, 30)
        self.excitationWavelengthSlider.setRange(400,700)
        self.excitationWavelengthSlider.setValue(int(DEFAULT_PARAMETERS["Excitation Wavelength"]))
        self.excitationWavelengthSlider.setSingleStep(1)
        self.excitationWavelengthSlider.setPageStep(20)        
        self.excitationWavelengthSpinBox = QtWidgets.QDoubleSpinBox(thistab)
        self.excitationWavelengthSpinBox.setGeometry(230,10,70,30)
        self.excitationWavelengthSpinBox.setMinimum(400)
        self.excitationWavelengthSpinBox.setMaximum(700)
        self.excitationWavelengthSpinBox.setSingleStep(1)
        self.excitationWavelengthSpinBox.setDecimals(2)
        self.excitationWavelengthSpinBox.setValue(DEFAULT_PARAMETERS["Excitation Wavelength"])
        self.excitationWavelengthSlider.valueChanged.connect(self.excite_slider)
        self.excitationWavelengthSpinBox.valueChanged.connect(self.excite_spinbox)
        self.excitationWavelengthSpinBox.setStyleSheet(PARAM_STYLE_B)

        self.sio2thicknessLabel = QtWidgets.QLabel("SiO2 (nm):", thistab)
        self.sio2thicknessLabel.setGeometry(5, 10+40, 60, 30)
        self.sio2thicknessSlider = QtWidgets.QSlider(Qt.Horizontal, thistab)
        self.sio2thicknessSlider.setGeometry(65, 10+40, 160, 30)
        self.sio2thicknessSlider.setRange(200000,3000000)
        self.sio2thicknessSlider.setValue(int(DEFAULT_PARAMETERS["SiO2 Thickness"]*1000))
        self.sio2thicknessSlider.setSingleStep(1)
        self.sio2thicknessSlider.setPageStep(20)        
        self.sio2thicknessSpinBox = QtWidgets.QDoubleSpinBox(thistab)
        self.sio2thicknessSpinBox.setGeometry(230,10+40,70,30)
        self.sio2thicknessSpinBox.setMinimum(1.)
        self.sio2thicknessSpinBox.setDecimals(4)
        self.sio2thicknessSpinBox.setMaximum(3000.)
        self.sio2thicknessSpinBox.setSingleStep(0.01)
        self.sio2thicknessSpinBox.setValue(DEFAULT_PARAMETERS["SiO2 Thickness"])       
        self.sio2thicknessSlider.valueChanged.connect(self.sio2thickness_slider)
        self.sio2thicknessSpinBox.valueChanged.connect(self.sio2thickness_spinbox)
        self.sio2thicknessSpinBox.setStyleSheet(PARAM_STYLE_B)
        
        self.sio2indexLabel = QtWidgets.QLabel("SiO2-R.I.:", thistab)
        self.sio2indexLabel.setGeometry(5, 10+40+40, 60, 30)
        self.sio2indexSlider = QtWidgets.QSlider(Qt.Horizontal, thistab)
        self.sio2indexSlider.setGeometry(65, 10+40+40, 160, 30)
        self.sio2indexSlider.setRange(1000,2000)
        self.sio2indexSlider.setValue(int(DEFAULT_PARAMETERS["SiO2 refractive index"]*1000))
        self.sio2indexSlider.setSingleStep(1)
        self.sio2indexSlider.setPageStep(20)        
        self.sio2indexSpinBox = QtWidgets.QDoubleSpinBox(thistab)
        self.sio2indexSpinBox.setGeometry(230,10+40+40,70,30)
        self.sio2indexSpinBox.setMinimum(1.00)
        self.sio2indexSpinBox.setMaximum(2.00)
        self.sio2indexSpinBox.setDecimals(6)
        self.sio2indexSpinBox.setSingleStep(0.01)
        self.sio2indexSpinBox.setValue(DEFAULT_PARAMETERS["SiO2 refractive index"])
        self.sio2indexSlider.valueChanged.connect(self.sio2index_slider)
        self.sio2indexSpinBox.valueChanged.connect(self.sio2index_spinbox)
        self.sio2indexSpinBox.setStyleSheet(PARAM_STYLE_B)

        self.bufferindexLabel = QtWidgets.QLabel("Buffer-R.I.:", thistab)
        self.bufferindexLabel.setGeometry(5, 10+40+40*2, 60, 30)
        self.bufferindexSlider = QtWidgets.QSlider(Qt.Horizontal, thistab)
        self.bufferindexSlider.setGeometry(65, 10+40+40*2, 160, 30)
        self.bufferindexSlider.setRange(1000,2000)
        self.bufferindexSlider.setValue(int(DEFAULT_PARAMETERS["Buffer refractive index"]*1000))
        self.bufferindexSlider.setSingleStep(1)
        self.bufferindexSlider.setPageStep(20)        
        self.bufferindexSpinBox = QtWidgets.QDoubleSpinBox(thistab)
        self.bufferindexSpinBox.setGeometry(230,10+40+40*2,70,30)
        self.bufferindexSpinBox.setMinimum(1.00)
        self.bufferindexSpinBox.setMaximum(2.00)
        self.bufferindexSpinBox.setDecimals(6)
        self.bufferindexSpinBox.setSingleStep(0.01)
        self.bufferindexSpinBox.setValue(DEFAULT_PARAMETERS["Buffer refractive index"])
        self.bufferindexSlider.valueChanged.connect(self.bufferindex_slider)
        self.bufferindexSpinBox.valueChanged.connect(self.bufferindex_spinbox)
        self.bufferindexSpinBox.setStyleSheet(PARAM_STYLE_B)

        self.siindexLabel = QtWidgets.QLabel("Si-R.I.:", thistab)
        self.siindexLabel.setGeometry(5, 10+40+40*3, 60, 30)
        self.siindexSlider = QtWidgets.QSlider(Qt.Horizontal, thistab)
        self.siindexSlider.setGeometry(65, 10+40+40*3, 160, 30)
        self.siindexSlider.setRange(3000,5000)
        self.siindexSlider.setValue(int(DEFAULT_PARAMETERS["Si refractive index"]*1000))
        self.siindexSlider.setSingleStep(1)
        self.siindexSlider.setPageStep(20)        
        self.siindexSpinBox = QtWidgets.QDoubleSpinBox(thistab)
        self.siindexSpinBox.setGeometry(230,10+40+40*3,70,30)
        self.siindexSpinBox.setMinimum(3.00)
        self.siindexSpinBox.setMaximum(5.00)
        self.siindexSpinBox.setDecimals(6)
        self.siindexSpinBox.setSingleStep(0.01)
        self.siindexSpinBox.setValue(DEFAULT_PARAMETERS["Si refractive index"])
        self.siindexSlider.valueChanged.connect(self.siindex_slider)
        self.siindexSpinBox.valueChanged.connect(self.siindex_spinbox)        
        self.siindexSpinBox.setStyleSheet(PARAM_STYLE_B)
        
    def init_parameters(self):
        try:
            self.nbuffer= DEFAULT_PARAMETERS["Buffer refractive index"]         
            self.nsio2 = DEFAULT_PARAMETERS["SiO2 refractive index"] 
            self.nspacer = DEFAULT_PARAMETERS["SiO2 refractive index"]
            self.nsi = DEFAULT_PARAMETERS["Si refractive index"]
            self.wavelength = DEFAULT_PARAMETERS["Excitation Wavelength"]
            self.spacerthickness = DEFAULT_PARAMETERS["SiO2 Thickness"]
            self.maximumZ = DEFAULT_PARAMETERS["Maximum Z"]
            self.maximumAngle = DEFAULT_PARAMETERS["Maximum Incidence Angle"]
            self.angleInterval = DEFAULT_PARAMETERS["Incidence Angle Interval"]
            self.zInterval = DEFAULT_PARAMETERS["Z Interval"]
            self.specifiedZ = 100
            self.specifiedA = 0
            self.cache = None
            self.matplotlibcolormap = MATPLOTLIB_CTABLE[self.pcolorDroplist.currentIndex()]        
            self.thetabdeg = [] # np.float64(np.arange(self.numanglepoints)*self.angleInterval)
            self.cameraOffset = DEFAULT_PARAMETERS["Camera Offset"]
            self.pixelsizenm = DEFAULT_PARAMETERS["Pixel Size"] 
            self.guessIteration = DEFAULT_PARAMETERS["Guess Iteration"]
            self.guessmaximum = DEFAULT_PARAMETERS["Guess Maximum"]
            self.minExptAngle = DEFAULT_PARAMETERS["Min. Expt. Angle"]
            self.exptAngleOffset = DEFAULT_PARAMETERS["Expt. Angle Offset"]
            self.exptAngleInterval = DEFAULT_PARAMETERS["Expt. Angle Int."]
            self.exptNumberOfFrames = DEFAULT_PARAMETERS["# Angle-points"]
            self.nglass = DEFAULT_PARAMETERS["Glass Index"]
            self.exptThetabdeg = np.float64(np.arange(self.exptNumberOfFrames)*self.exptAngleInterval+self.minExptAngle+self.exptAngleOffset) 
            
            self.maximumWavelength = DEFAULT_PARAMETERS['Maximum Wavelength']
            self.minimumWavelength = DEFAULT_PARAMETERS['Minimum Wavelength']
            self.wavelengthInterval = DEFAULT_PARAMETERS['Wavelength Interval']
        except:
            print('Some parameters not set properly...')
            
    def autoRI(self):
        self.wavelength = self.excitationWavelengthSpinBox.value()   
        self.nbuffer = refractiveIndices.getIndex('H2O', self.wavelength)
        self.nsio2 = refractiveIndices.getIndex('SiO2', self.wavelength)
        self.nsi =  refractiveIndices.getIndex('Si', self.wavelength)
        print('Auto-interpolated refractive indices @', self.wavelength)
        print('Si:', self.nsi)
        print('SiO2:',self.nsio2)
        print('H2O:',self.nbuffer)
        self.nspacer = self.nsio2
        
        self.bufferindexSpinBox.setValue(self.nbuffer)
        self.bufferindexSlider.setValue(int(self.nbuffer*1000))      
           
        self.sio2indexSpinBox.setValue(self.nsio2)
        self.sio2indexSlider.setValue(int(self.nsio2*1000))        
                            
        self.siindexSpinBox.setValue(self.nsi)
        self.siindexSlider.setValue(int(self.nsi*1000))      
        
    def chooseLUT(self, value):   
        self.matplotlibcolormap = MATPLOTLIB_CTABLE[value]
        
    def parametersUpdate(self):
        try:
            self.nbuffer= self.bufferindexSpinBox.value()        
            self.nsi = self.siindexSpinBox.value()
            self.nspacer = self.sio2indexSpinBox.value()   
            self.nsio2 = self.sio2indexSpinBox.value()   
            self.wavelength = self.excitationWavelengthSpinBox.value()
            self.spacerthickness = self.sio2thicknessSpinBox.value()
            self.maximumZ = self.maxZSpinBox.value()
            self.maximumAngle = self.maxAngleSpinBox.value()
            self.angleInterval = self.intervalSpinBox.value()
            self.zInterval = self.zIntervalSpinBox.value()
            self.specifiedZ = self.specificZSpinBox.value()
            self.numanglepoints = np.floor(self.maximumAngle/self.angleInterval)+1        
            self.thetabdeg = np.float64(np.arange(self.numanglepoints)*self.angleInterval)
        except:
           print('Not all parameters set properly...')
           
    def simulateFluorophoreZ(self):
        return lib.simulateFluorophore(self, extra=self.plotOptionDroplist.currentIndex(), size = 100, fwhm = 35, center=None, A = self.specificASpinBox.value(), colormap = self.matplotlibcolormap)
    
    def waveSimulate2D(self):
        return lib.fieldSimulate2D(self, extra = self.plotOptionDroplist.currentIndex(),numba = None, fixedTheta = 1, thetadeg = self.specificASpinBox.value())
    
    def fieldSimulate2D(self):
        return lib.fieldSimulate2D(self, extra = self.plotOptionDroplist.currentIndex(), transpose = 1)
        
    def fieldSimulate3D(self):
        return lib.fieldSimulate3D(self, extra = self.plotOptionDroplist.currentIndex())
        
    def fieldSimulate1D(self):
        return lib.fieldSimulate1D(self, extra = self.plotOptionDroplist.currentIndex())
        
    def fieldSimulate1A(self):
        return lib.fieldSimulate1A(self, extra = self.plotOptionDroplist.currentIndex())
        
    def exportCalib(self):
        save_dataframe_csv(self,filename = None, dataframe = pd.DataFrame(self.fieldSimulateMulti(getCalib=1)), mode = 'SGI',  
                           prompt = "Save SGiPALM calibration to *SGi.csv file", filter = "*SGi.csv" )

    def fieldSimulateMulti(self, *, getCalib = 0, as_dict = 1):  
        if hasattr(self,'plotOptionDroplist'):
            extra = self.plotOptionDroplist.currentIndex()
        else:
            extra = 1
        thetext = self.multiAngleLineEdit.text()
        angleval =  [int(_) for _ in thetext.split(',')]
        plots = []
        fits = []
        for _ in angleval:
            x,F,_fit = lib.fieldSimulate1A(self, extra = 0, A = _, style = 'r')
            plots.append(F)
            fits.append(get_positive(_fit))
        
        if extra:
            figure, axes = plt.subplots(1,1)
            for k in plots:
                axes.plot(x,k)
            axes.set_xlim(0,np.max(x))
            axes.set_ylim(0,np.max(plots)*1.1)
            axes.set_position([0.08,0.08,0.9,0.9])
            axes.set_xlabel('Z(nm)')
            axes.set_ylabel('a.u.')
            axes.grid(True)
            plt.show()
        else:
            self.axes.clear()
            for k in plots:
                self.axes.plot(x,k)
            self.axes.set_xlim(0,np.max(x))
            self.axes.set_ylim(0,np.max(plots)*1.1)
            self.axes.set_position([0.08,0.08,0.9,0.9])
            self.axes.set_xlabel('Z(nm)')
            self.axes.set_ylabel('a.u.')
            self.axes.grid(True)
            self.canvas.draw()
            self.canvas.show()
            
        if getCalib:
            if as_dict:
                return [dict(offset=_[0],amplitude=_[1],period=_[2],phase = _[3]) for _ in fits]
            else:
                return [[_[0],_[1],_[3],_[2]] for _ in fits]
        else:
            return x,plots
    
    def plotCRLBz_search(self):
        self.plotCRLBz(minimize=1)
        
    def plotCRLBz_variation(self):
        self.plotCRLBz(variation=3)
            
    def plotCRLBz(self, *, minimize = 0, variation = 0):
        extra = self.plotOptionDroplist.currentIndex()
        thetext = self.multiAngleLineEdit.text()
        angleval =  [int(_) for _ in thetext.split(',')]
        self.parametersUpdate()
        
        Ival = [int(_) for _ in self.IsigLineEdit.text().split(',')]       
        bval = [int(_) for _ in self.bLineEdit.text().split(',')] 
        if len(Ival) != len(bval):
            print('Unmatched values for I and b...')
            exit
        
        zi = 0
        zpoints = int((self.maximumZ-zi)/self.zInterval)+1
        zarray = np.linspace(zi,self.maximumZ,num=zpoints)

        if variation:
            vv = [ i-variation for i in range (1+2*int(variation))]
            n_angle = len(angleval)
           
            for index,j in enumerate(angleval):
                figure, axesv = plt.subplots(1,1)
                for _i,k in enumerate(vv):
                    aa = angleval.copy()
                    aa[index]  = angleval[index]+k
                    _z,results = lib.CRLB_Plot(plot= 0, sqrt=1,  PSNR = Ival[0]/bval[0], z=zarray, zrange = None, background = bval[0], 
                              wavelength =self.wavelength, dox = self.spacerthickness, thetabarray =aa,nbuffer=self.nbuffer,nspacer=self.nspacer,nsi=self.nsi )
                    axesv.plot(_z,results, linewidth =k)
                _z,results = lib.CRLB_Plot(plot= 0, sqrt=1,  PSNR = Ival[0]/bval[0], z=zarray, zrange = None, background = bval[0], 
                           wavelength =self.wavelength, dox = self.spacerthickness, thetabarray =angleval,nbuffer=self.nbuffer,nspacer=self.nspacer,nsi=self.nsi )
                axesv.plot(_z,results, linewidth =3, color = 'black', alpha = .5)     
                axesv.set_position([0.08,0.08,0.9,0.9])
                axesv.set_xlabel('Z(nm)')
                axesv.set_ylabel('nm')  
            return
        
        if minimize:
             args = dict(plot=0, sqrt=1,  PSNR = self.IsigSpinBox.value()/self.bSpinBox.value(), z=zarray, zrange = None, verbose = 1, background = self.bSpinBox.value(), 
                         wavelength =self.wavelength, dox = self.spacerthickness,  nbuffer=self.nbuffer,nspacer=self.nspacer,nsi=self.nsi)
        
             print('Minization by Nelder-Mead; initial guess:', angleval)
             res = minimize(lib.CRLB_func, angleval, method='Nelder-Mead', tol=1e-6)
             print('Minimization results:', res.x)
             _z,results_ = lib.CRLB_Plot(plot= 0, sqrt=1,  PSNR = self.IsigSpinBox.value()/self.bSpinBox.value(), z=zarray, zrange = None, background = self.bSpinBox.value(), 
                           wavelength =self.wavelength, dox = self.spacerthickness, thetabarray =res.x,nbuffer=self.nbuffer,nspacer=self.nspacer,nsi=self.nsi )
             axes.plot(_z,results_)
             print(np.sum(results_))
             return

        figure, axes = plt.subplots(1,1)
        for i,b in zip(Ival,bval):
            _z,results = lib.CRLB_Plot(plot= 0, sqrt=1,  PSNR = i/b, z=zarray, zrange = None, background = b, 
                      wavelength =self.wavelength, dox = self.spacerthickness, thetabarray =angleval,nbuffer=self.nbuffer,
                      nspacer=self.nspacer,nsi=self.nsi )
            axes.plot(_z,results)
            print(np.sum(results))
        axes.set_xlim(0,np.max(_z))
        axes.set_ylim(0,np.max(results)*1.1)
        axes.set_position([0.08,0.08,0.9,0.9])
        axes.set_xlabel('Z(nm)')
        axes.set_ylabel('nm')
        
    def clearScreen(self):
        self.reset_canvas()
        
    def closeEvent(self, event):
        print("Terminating...")
        # QtWidgets.qApp.closeAllWindows()

    def reset_canvas(self):
        self.canvas = PlotCanvas(self, width=5, height=4)
        self.canvas.setGeometry(self.viewportPosition[0], self.viewportPosition[1], self.viewportSize[0], self.viewportSize[1])
        self.axes = self.canvas.figure.add_subplot(111)
        self.canvas.show()

    def exptangleinterval_spinbox(self, value):
        self.expAngleIntervalSlider.setValue(int(value)); self.exptAngleInterval = value
    def exptangleinterval_slider(self,value):
        self.expAngleIntervalSpinBox.setValue(value); self.exptAngleInterval = value
    def numframes_spinbox(self, value):
        self.numFramesSlider.setValue(int(value)); self.exptNumberOfFrames = value
    def numframes_slider(self, value):
        self.numFramesSpinBox.setValue(value); self.exptNumberOfFrames = value
    def minexptangle_spinbox(self, value):
        self.minExptAngle = value
    def angleoffset_spinbox(self, value):
        self.exptAngleOffset = value
    def bufferindex_slider(self, value):
        self.bufferindexSpinBox.setValue(value/1000);        self.nbuffer = value
    def bufferindex_spinbox(self, value):
        self.bufferindexSlider.setValue(int(value*1000));        self.nbuffer = value
    def siindex_slider(self, value):
        self.siindexSpinBox.setValue(value/1000);        self.nsi = value
    def siindex_spinbox(self, value):
        self.siindexSlider.setValue(int(value*1000));        self.nsi = value
    def specificz_slider(self, value):
        self.specificZSpinBox.setValue(value);        self.specifiedZ = value
    def specificz_spinbox(self, value):
        self.specificZSlider.setValue(int(value));        self.specifiedZ = value
    def specificA_slider(self, value):
        self.specificASpinBox.setValue(value);        self.specifiedA = value
    def specificA_spinbox(self, value):
        self.specificASlider.setValue(int(value));        self.specifiedA = value
        
    def objectz_slider(self, value):
        self.ObjectZSpinBox.setValue(value)      
    def objectz_spinbox(self, value):
        self.ObjectZSlider.setValue(int(value))     
    def maxbleach_slider(self, value):
        self.maxBleachSpinBox.setValue(value)
    def maxbleach_spinbox(self,value):
        self.maxBleachSlider.setValue(int(value))
        
    def zinterval_slider(self, value):
        self.zIntervalSpinBox.setValue(value);        self.zInterval = value
    def zinterval_spinbox(self, value):
        self.zIntervalSlider.setValue(int(value));        self.zInterval = value
    def maxangle_slider(self, value):
        self.maxAngleSpinBox.setValue(value);        self.maximumAngle = value
    def maxangle_spinbox(self, value):
        self.maxAngleSlider.setValue(int(value));        self.maximumAngle = value
    def maxz_slider(self, value):
        self.maxZSpinBox.setValue(value);        self.maximumZ = value
    def maxz_spinbox(self, value):
        self.maxZSlider.setValue(int(value));        self.maximumZ = value
    def interval_slider(self, value):
        self.intervalSpinBox.setValue(value);        self.angleInterval = value
    def interval_spinbox(self, value):
        self.intervalSlider.setValue(int(value));        self.angleInterval = value
    def minlambda_spinbox(self,value):
        self.minLambdaSlider.setValue(int(value)); self.minimumWavelength = value
    def minlambda_slider(self,value): 
        self.minLambdaSpinBox.setValue(value); self.minimumWavelength = value
    def l_interval_slider(self,value): 
        self.l_intervalSpinBox.setValue(value); self.wavelengthInterval = value
    def l_interval_spinbox(self,value): 
        self.l_intervalSlider.setValue(int(value)); self.wavelengthInterval = value
    def maxlambda_slider(self,value): 
        self.maxLambdaSpinBox.setValue(value); self.maximumWavelength = value
    def maxlambda_spinbox(self,value):
        self.maxLambdaSlider.setValue(int(value)); self.maximumWavelength = value
        
    def excite_slider(self,value):
        self.excitationWavelengthSpinBox.setValue(value);        self.wavelength = value
    def excite_spinbox(self, value):
        self.excitationWavelengthSlider.setValue(int(value));        self.wavelength = value
    def sio2thickness_slider(self,value):
        self.sio2thicknessSpinBox.setValue(value/1000);        self.spacerthickness = value    
        # print(self.spacerthickness)
    def sio2thickness_spinbox(self,value):
        self.sio2thicknessSlider.setValue(int(value*1000));        self.spacerthickness = value
        # print(self.spacerthickness)
    def sio2index_spinbox(self,value):
        self.sio2indexSlider.setValue(int(value*1000));        self.nsio2 = value
    def sio2index_slider(self,value):
        self.sio2indexSpinBox.setValue(value/1000);        self.nsio2 = value

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)        

def main():
    app = QtWidgets.QApplication(sys.argv)    
    FONTSIZE = 7
    desktop = app.desktop()
    screen_geometry = desktop.screenGeometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()
    print("Screen resolution:", screen_width, "x", screen_height)
    if screen_width >1920:
        print('Please set screen resolution to FHD or FHD+ for best results...')
    print('Font Size:', FONTSIZE)
    app.setAttribute(Qt.AA_DisableHighDpiScaling)    
    app.setFont(QtGui.QFont("Arial", FONTSIZE))
    
    window2 = SGI()
    window2.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()



