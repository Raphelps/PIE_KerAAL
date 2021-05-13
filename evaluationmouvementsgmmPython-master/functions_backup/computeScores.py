import numpy as np
from functions.Nodes import Snode
from functions.scoreToPercentage import scoreToPercentage


def computeScores(m_cuts, m_cutsKP, Lglobal, Lbodypart, Ljoints, seuils, minseuils):
    """
    :param model: corresponding trained model to obtain segmentation results
    :param Lglobal: likelihood
    :param Lbodypart: likelihood
    :param Ljoints: likelihood
    :param seuils:  the threshold vector of length 6 corresponding to global, global ori or
                    pos, body part, body part ori or pos, joint and joint ori or pos
    :param minseuils: the minimal threshold vector of length 6 corresponding to global, global ori or
                    pos, body part, body part ori or pos, joint and joint ori or pos.

    :return:  Sglobal structure (Sglobal, Sori, Spos)
    :return:  Sbodypart scores per body parts (global, pos, ori)
    :return:  Sjoints scores per joints (global, pos, ori per joints, same size as dataTest)
    """
    Sglobal = {}
    Sbodypart = {}
    Sjoints = {}

    # Global
    for key in Lglobal:
        data = Lglobal[key].data
        global_ = [np.mean(data)]
        perSegment = [np.mean(data[:m_cuts[0]])]
        perSegmentKP = [np.mean(data[:m_cutsKP[0]])]
        for c in range(m_cuts.size-1):
            perSegment.append(np.mean(data[m_cuts[c]:m_cuts[c + 1]]))
        for c in range(m_cutsKP.size - 1):
            perSegmentKP.append(np.mean(data[m_cutsKP[c]:m_cutsKP[c + 1]]))
        perSegment.append(np.mean(data[m_cuts[-1]:]))
        perSegmentKP.append(np.mean(data[m_cutsKP[-1]:]))

        if key == 'Global':
            Sglobal[key] = Snode(scoreToPercentage(global_, seuils[0], minseuils[0]),
                                 scoreToPercentage(perSegment, seuils[0], minseuils[0]),
                                 scoreToPercentage(perSegmentKP, seuils[0], minseuils[0])
                                 )
        else :
            Sglobal[key] = Snode(scoreToPercentage(global_, seuils[1], minseuils[1]),
                             scoreToPercentage(perSegment, seuils[1], minseuils[1]),
                             scoreToPercentage(perSegmentKP, seuils[1], minseuils[1])
                             )

    # Body Parts
    for key in Lbodypart:
        tmp = {}
        for item in Lbodypart[key]:
            data = Lbodypart[key][item].data
            global_ = [np.mean(data)]
            perSegment = [np.mean(data[:m_cuts[0]])]
            perSegmentKP = [np.mean(data[:m_cutsKP[0]])]
            for c in range(m_cuts.size - 1):
                perSegment.append(np.mean(data[m_cuts[c]:m_cuts[c + 1]]))
            for c in range(m_cutsKP.size - 1):
                perSegmentKP.append(np.mean(data[m_cutsKP[c]:m_cutsKP[c + 1]]))
            perSegment.append(np.mean(data[m_cuts[-1]:]))
            perSegmentKP.append(np.mean(data[m_cutsKP[-1]:]))

            if item == 'Left Arm Global' or item == 'Right Arm Global':
                tmp[item] = Snode(scoreToPercentage(global_, seuils[2], minseuils[2]),
                                     scoreToPercentage(perSegment, seuils[2], minseuils[2]),
                                     scoreToPercentage(perSegmentKP, seuils[2], minseuils[2])
                                     )
            else:
                tmp[item] = Snode(scoreToPercentage(global_, seuils[3], minseuils[3]),
                                     scoreToPercentage(perSegment, seuils[3], minseuils[3]),
                                     scoreToPercentage(perSegmentKP, seuils[3], minseuils[3])
                                     )
        Sbodypart[key] = tmp.copy()

    # joints
    for key in Ljoints:
        tmp = {}
        for item in Ljoints[key]:
            data = Ljoints[key][item].data
            global_ = [np.mean(data)]
            perSegment = [np.mean(data[:m_cuts[0]])]
            perSegmentKP = [np.mean(data[:m_cutsKP[0]])]
            for c in range(m_cuts.size - 1):
                perSegment.append(np.mean(data[m_cuts[c]:m_cuts[c + 1]]))
            for c in range(m_cutsKP.size - 1):
                perSegmentKP.append(np.mean(data[m_cutsKP[c]:m_cutsKP[c + 1]]))
            perSegment.append(np.mean(data[m_cuts[-1]:]))
            perSegmentKP.append(np.mean(data[m_cutsKP[-1]:]))

            if item == 'lElbow Global' or item == 'lWrist Global' or item == 'lShoulder Global' or \
                    item == 'rElbow Global' or item == 'rWrist Global' or item == 'rShoulder Global':
                tmp[item] = Snode(scoreToPercentage(global_, seuils[4], minseuils[4]),
                                     scoreToPercentage(perSegment, seuils[4], minseuils[4]),
                                     scoreToPercentage(perSegmentKP, seuils[4], minseuils[4])
                                     )
            else:
                tmp[item] = Snode(scoreToPercentage(global_, seuils[5], minseuils[5]),
                                     scoreToPercentage(perSegment, seuils[5], minseuils[5]),
                                     scoreToPercentage(perSegmentKP, seuils[5], minseuils[5])
                                     )
        Sjoints[key] = tmp.copy()

    return Sglobal, Sbodypart, Sjoints
