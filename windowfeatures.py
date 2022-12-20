import numpy as np
import re
import pandas as pd

class createWindowData(object):
    _daa = dict()
    _sequence = ""
    _yIons = []
    _bIons = []
    _yIonsNorm = []
    _bIonsNorm = []
    _matrix = []

    def __init__(self, sequence, ions, intensities):
        self._sequence = sequence
        # Creating Amino Matrix
        aa = "-ARNDCQEGHILKMFPSTWYV"
        MAA = np.identity(len(aa))
        self._daa = dict()
        for index, arrayAA in enumerate(MAA):
            self._daa[aa[index]] = arrayAA
 
        self.yIons = [0] * len(sequence)
        self.bIons = [0] * len(sequence)
        lstions = ions.split(";")
        lstintensities = intensities.split(";")

        yIonsreg = re.compile('^y[0-9]+$')
        bIonsreg = re.compile('^b[0-9]+$')

        self._yIons = [0] * len(sequence)
        self._bIons = [0] * len(sequence)


        for index, ion in enumerate(lstions):
            # print ion , lstintensities[index]
            if yIonsreg.match(ion):  #
                self._yIons[int(ion.split("y")[1])] = float(lstintensities[index])
            if bIonsreg.match(ion):
                self._bIons[int(ion.split("b")[1])] = float(lstintensities[index])
        list_b_ions = self._bIons[1:]
        list_b_ions.append(0)
        list_y_ions = self._yIons[::-1]
        self._bIons = list_b_ions
        self._yIons = list_y_ions

        self._matrix = []

    def GenerateMatrix(self, size):
        dictFeature = dict()
        order = []
        size = int(size)
        for index in range(((size) // 2) - 1):
            key = "Sj-" + str(index + 1)
            dictFeature[key] = ["?"] * 20
            order.append((index + 1) * -1)
        ##############
        dictFeature["Sj"] = ["?"] * 20
        ##############
        for index in range((size) // 2):
            key = "Sj+" + str(index + 1)
            dictFeature[key] = ["?"] * 20
            order.append(index + 1)
            ##########################
        dictFeature["Sj+1"] = ["?"] * 20
        dictFeature["Sj+2"] = ["?"] * 20
        ############################
        dictFeature["S1"] = ["?"] * 20
        dictFeature["SN"] = ["?"] * 20
        dictFeature["length"] = "?"
        dictFeature["Dist-1"] = "?"
        dictFeature["Dist-N"] = "?"
        Sj_Positive = re.compile('^Sj\+')
        Sj_Negative = re.compile('^Sj-')
        order.append(0)
        order = sorted(order)

        for index, aa in enumerate(self._sequence):
            for k in dictFeature:
                val = 0
                if Sj_Negative.match(k):
                    val = int(k[3:]) * -1

                if Sj_Positive.match(k):
                    val = int(k[3:])

                if (index + val) < 0 or (index + val) >= len(self._sequence):
                    dictFeature[k] = self._daa["-"]
                else:
                    dictFeature[k] = self._daa[self._sequence[index + val]]
            # dictFeature["Sj"]
            dictFeature["Sj"] = self._daa[self._sequence[index]]
            # dictFeature["S1"]
            dictFeature["S1"] = self._daa[self._sequence[0]]
            # dictFeature["SN"]
            dictFeature["SN"] = self._daa[self._sequence[len(self._sequence) - 1]]
            # dictFeature["length"]
            dictFeature["length"] = len(self._sequence)
            # dictFeature["Dist-1"]="?"
            dictFeature["Dist-1"] = index
            # dictFeature["Dist-N"]="?"
            dictFeature["Dist-N"] = len(self._sequence) - index - 1
            # print aa
            # print dictFeature
            col = []
            for i in order:
                if (i < 0):
                    col += list(dictFeature["Sj" + str(i)])
                if (i == 0):
                    col += list(dictFeature["Sj"])
                if (i > 0):
                    col += list(dictFeature["Sj+" + str(i)])

            col += list(dictFeature["S1"]) + list(dictFeature["SN"]) + [dictFeature["length"]] + [
                dictFeature["Dist-1"] + 1] + [dictFeature["Dist-N"]]
            res = [int(i) for i in col]
            self._matrix.append(res)

    def nomalizeIones(self):
        self._yIonsNorm = [float(i) / max(self._yIons + self._bIons) for i in self._yIons]
        self._bIonsNorm = [float(i) / max(self._yIons + self._bIons) for i in self._bIons]

    def GenerateDataset(self, removeZeros):
        self.nomalizeIones()
        data = {'featureMatrix': [], 'target_Y': [], 'target_B': []}

        for index, line in enumerate(self._matrix):
            if removeZeros == True:
                if (self._yIonsNorm[index] != 0):
                    data['featureMatrix'].append(line)
                    # data['target_Y'].append(("%.2f" %  self._yIonsNorm[index]))
                    data['target_Y'].append((self._yIonsNorm[index]))

                if (self._bIonsNorm[index] != 0):
                    # data['featureMatrix_B'].append(line)
                    data['target_B'].append((self._bIonsNorm[index]))
            else:
                data['featureMatrix'].append(line)
                data['target_Y'].append((self._yIonsNorm[index]))
                # data['featureMatrix_B'].append(line)
                data['target_B'].append((self._bIonsNorm[index]))
        data['target_Y'] = list(map(float, data['target_Y']))
        data['target_B'] = list(map(float, data['target_B']))
        # Correction Bug
        data['featureMatrix'] = data['featureMatrix'][:-1]
        data['target_Y'] = data['target_Y'][:-1]
        data['target_B'] = data['target_B'][:-1]
        return data