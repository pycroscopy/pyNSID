import os, sys
sys.path.append('../../../sidpy/')
import sidpy as sid
import sidpy.io.interface_utils as iu
import file_tools_nsid as ft
import time


fp = open(ft.config_path+'\\path.txt', 'r')
path = fp.read()
print(path)
fp.close()

import sidpy.io.interface_utils as iu
filename = iu.openfile_dialog(file_types="pyNSID (*.hf5);;All file (*)", file_path=path)

print(filename)
try:
    from PyQt5 import QtGui, QtWidgets, QtCore
except:
    pass
progress_bar = iu.ProgressDialog()
for count in range(100):
    progress_bar.set_value(count)
progress_bar.close()

