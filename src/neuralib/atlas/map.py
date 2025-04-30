from typing import Literal

from neuralib.typing import Series
from .typing import Area, TreeLevel

__all__ = [
    'NUM_MERGE_LAYER',
    'DEFAULT_FAMILY_DICT',
    'ALLEN_FAMILY_TYPE',
    'merge_until_level'
]

NUM_MERGE_LAYER = 5

# =========================== #
# LEVEL 4: Mainly Merge layer #
# =========================== #

MERGE_LEVEL_LAYER: TreeLevel = 4
MERGE_REGION_LV4 = {
    # VS
    'VL': ['SEZ', 'chpl'],
    'V4': ['V4r'],

    # CB
    'AN': ['ANcr1', 'ANcr2'],
    'CENT': ['CENT*'],
    'CUL': ['CUL*'],

    # HPF
    'ENTl': ['ENTl*'],
    'ENTm': ['ENTm*'],

    # Isocortex
    'ACAd': ['ACAd*'],
    'ACAv': ['ACAv*'],
    'AId': ['AId*'],
    'AIp': ['AIp*'],
    'AIv': ['AIv*'],
    'AUDd': ['AUDd*'],
    'AUDp': ['AUDp*'],
    'AUDpo': ['AUDpo*'],
    'AUDv': ['AUDv*'],
    'ECT': ['ECT*'],
    'FRP': ['FRP*'],
    'GU': ['GU*'],
    'ILA': ['ILA*'],
    'MOp': ['MOp*'],
    'MOs': ['MOs*'],
    'ORBl': ['ORBl*'],
    'ORBm': ['ORBm*'],
    'ORBvl': ['ORBvl*'],
    'PERI': ['PERI*'],
    'PL': ['PL*'],
    'VISa': ['VISa1', 'VISa2/3', 'VISa4', 'VISa5', 'VISa6a', 'VISa6b'],  # NOTE PTLp child
    'VISrl': ['VISrl*'],
    'RSPagl': ['RSPagl*'],
    'RSPd': ['RSPd*'],
    'RSPv': ['RSPv*'],
    'SSs-bfd': ['SSs-bfd*'],
    'SSp-ll': ['SSp-ll*'],
    'SSp-m': ['SSp-m*'],
    'SSp-n': ['SSp-n*'],
    'SSp-tr': ['SSp-tr*'],
    'SSp-ul': ['SSp-ul*'],
    'SSp-un': ['SSp-un*'],
    'SSs': ['SSs*'],
    'TEa': ['TEa*'],
    'VISal': ['VISal*'],
    'VISam': ['VISam*'],
    'VISl': ['VISl1', 'VISl2/3', 'VISl4', 'VISl5', 'VISl6a', 'VISl6b'],  # differentiate the VISli
    'VISli': ['VISli*'],
    'VISp': ['VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b'],  # differentiate  VISpl, VISpm
    'VISpl': ['VISpl*'],
    'VISpm': ['VISpm*'],
    'VISpor': ['VISpor*'],
    'VISC': ['VISC*'],

    # OLF
    'NLOT': ['NLOT*']

}

# ============================== #
# LEVEL 3: Mainly merge DV ML AP #
# ============================== #

MERGE_LEVEL_DVMLAP: TreeLevel = 3
MERGE_REGION_LV3 = {
    # HB
    'AMB': ['AMB*'],
    'LRN': ['LRN*'],
    'MDRN': ['MDRN*'],
    'PGRN': ['PGRN*'],
    'SOC': ['POR', 'SOC*'],

    # HY
    'MM': ['MMd', 'MMl', 'MMm', 'MMme', 'MMp',
           'Mmd', 'Mml', 'Mmm', 'Mmme', 'Mmp'],  # diff in ccf
    'TM': ['TM*'],

    # TH
    'AM': ['AM*'],
    'LGd': ['LGd-*'],
    'MG': ['MG*'],
    'SPF': ['SPF*'],
    'VP': ['VPL', 'VPLpc', 'VPM', 'VPMpc'],

    # HPF
    'ENT': ['ENT*'],
    'DG': ['DG*'],

    # Isocortex
    'ACA': ['ACA*'],
    'AI': ['AId', 'AIp', 'AIv'],
    'AUD': ['AUD*'],
    'ORB': ['ORBl', 'ORBm', 'ORBvl'],
    'SSp': ['SSp*'],
    'AOB': ['AOB*'],
    'COA': ['COA*'],
    'TT': ['TT*'],

    # CNU
    'PALc': ['BAC', 'BST'],
    'PALd': ['GPe', 'GPi'],
    'MSC': ['MS', 'NDB'],
    'PALv': ['MA', 'SI'],
    'LS': ['LSc', 'LSr', 'LSv'],
    'CEA': ['CEA*'],

    # MB
    'SCm': ['SCdg', 'SCdw', 'SCig', 'SCiw'],
    'IC': ['IC*'],
    'SCs': ['SCop', 'SCsg', 'SCzo'],

    # CTXsp
    'BLA': ['BLA*'],
    'BMA': ['BMA*'],
    'EP': ['EP*'],

}

# ========================= #
# LEVEL 2: Customized Merge #
# ========================= #

MERGE_LEVEL_C2: TreeLevel = 2
MERGE_REGION_LV2 = {

    # HPF
    'CA': ['CA1', 'CA2', 'CA3'],
    'SUB': ['ProS'],  # might not correct, converge due to the area not in cellatlas (Prosubiculum)

    # Isocortex
    'SS': ['SS*'],
    'VIS': ['VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl', 'VISpm', 'VISpor'],
    'PTLp': ['VISa', 'VISrl'],
    'MO': ['MOp', 'MOs'],
    'RSP': ['RSP*'],

    # TH
    'ATN': ['AD', 'AM', 'AV', 'IAD', 'IAM', 'LD'],
    'EPI': ['LH', 'MH'],
    'GENv': ['IGL', 'IntG', 'LGv', 'SubG'],
    'ILM': ['CL', 'CM', 'PCN', 'PF', 'PIL', 'RH'],
    'LAT': ['Eth', 'LP', 'PO', 'POL', 'SGN'],
    'MED': ['IMD', 'MD', 'PR', 'SMT'],
    'MTN': ['PT', 'PVT', 'RE', 'Xi'],
    'GENd': ['LGd', 'MG'],
    'VENT': ['PoT', 'VAL', 'VM', 'VP'],

    # CNU
    'PALm': ['MSC', 'TRS'],
    'LSX': ['LS', 'SF', 'SH'],
    'STRd': ['CP'],
    'STRv': ['ACB', 'FS', 'OT'],
    'sAMY': ['AAA', 'BA', 'CEA', 'IA', 'MEA'],

    # HY
    'MBO': ['LM', 'MM', 'SUM', 'TM'],
    'ZI': ['FF'],

    # MB
    'IPN': ['IPA', 'IPC', 'IPDL', 'IPDM', 'IPI', 'IPL', 'IPR', 'IPRL'],

    # HB
    'PHY': ['NR', 'PRP'],
    'VNC': ['LAV', 'MV', 'SPIV', 'SUV'],
    'CN': ['DCO', 'VCO'],
    'DCN': ['CU', 'GR'],
    'PB': ['KF'],

    # CB
    'HEM': ['AN', 'COPY', 'FL', 'PFL', 'PRM', 'SIM'],
    'VERM': ['CENT', 'CUL', 'DEC', 'FOTU', 'LING', 'NOD', 'PYR', 'UVU'],

}

# ========================= #
# LEVEL 1: Customized Merge #
# ========================= #

MERGE_LEVEL_C1: TreeLevel = 1
MERGE_REGION_LV1 = {
    # VS
    'VS': ['AQ', 'V3', 'V4', 'VL'],

    # OLF
    'OLF': ['AOB', 'AON', 'COA', 'DP', 'MOB', 'NLOT', 'PAA', 'PIR', 'TR', 'TT'],

    # HPF
    'HIP': ['CA', 'DG', 'FC', 'IG'],
    'RHP': ['APr', 'ENT', 'HATA', 'PAR', 'POST', 'PRE', 'ProS', 'SUB'],

    # CNU
    'PAL': ['PALc', 'PALd', 'PALm', 'PALv'],
    'STR': ['LSX', 'STRd', 'STRv', 'sAMY'],

    # HY
    'LZ': ['LHA', 'LPO', 'PST', 'PSTN', 'PeF', 'RCH', 'STN', 'TU', 'ZI'],
    'MEZ': ['AHN', 'MBO', 'MPN', 'PH', 'PMd', 'PMv', 'PVHd', 'VMH'],
    'PVR': ['ADP', 'AVP', 'AVPV', 'DMH', 'MEPO', 'MPO', 'OV', 'PD', 'PS', 'PVp', 'PVpo', 'SBPV', 'SCH', 'SFO', 'VLPO',
            'VMPO'],
    'PVZ': ['ARH', 'ASO', 'PVH', 'PVa', 'PVi', 'SO'],

    # MB
    'PAG': ['INC', 'ND', 'PRC', 'Su3'],
    'PRT': ['APN', 'MPT', 'NOT', 'NPC', 'OP', 'PPT', 'RPF'],
    'RAmb': ['CLI', 'DR', 'IF', 'IPN', 'RL']

}

# ========================= #
# LEVEL 0: Customized Merge #
# ========================= #

MERGE_LEVEL_C0: TreeLevel = 0
MERGE_REGION_LV0 = {
    # HB
    'MY-mot': ['ACVII', 'AMB', 'DMX', 'GRN', 'ICB', 'IO', 'IRN', 'ISN', 'LIN', 'LRN', 'MARN', 'MDRN', 'PARN', 'PAS',
               'PGRN', 'PHY', 'PPY', 'VI', 'VII', 'VNC', 'XII', 'x', 'y'],
    'MY-sat': ['RM', 'RO', 'RPA'],
    'MY-sen': ['AP', 'CN', 'DCN', 'ECU', 'NTB', 'NTS', 'PA5', 'SPVC', 'SPVI', 'SPVO'],
    'P-mot': ['Acs5', 'B', 'DTN', 'I5', 'P5', 'PC5', 'PCG', 'PDTg', 'PG', 'PRNc', 'SG', 'SUT', 'TRN', 'V'],
    'P-sat': ['CS', 'LC', 'LDT', 'NI', 'PRNr', 'RPO', 'SLC', 'SLD'],
    'P-sen': ['NLL', 'PB', 'PSV', 'SOC'],

    #
    'HPF': ['HIP', 'RHP'],

    #
    'HY': ['LZ', 'ME', 'MEZ', 'PVR', 'PVZ'],

    #
    'TH': ['ATN', 'EPI', 'GENv', 'ILM', 'LAT', 'MED', 'MTN', 'RT',  # DORpm
           'GENd', 'PP', 'SPA', 'SPF', 'VENT'],  # DORsm

    # MB
    'MBmot': ['AT', 'CUN', 'DT', 'EW', 'III', 'IV', 'LT', 'MA3', 'MRN', 'MT', 'PAG',
              'PN', 'PRT', 'Pa4', 'RN', 'RR', 'SCm', 'SNr', 'VTA', 'VTN'],
    'MBsen': ['IC', 'MEV', 'NB', 'PBG', 'SAG', 'SCO', 'SCs'],
    'MBsta': ['PPN', 'RAmb', 'SNc'],

    # CB
    'CBN': ['DN', 'FN', 'IP', 'VeCB'],
    'CBX': ['HEM', 'VERM']

}

# ==================================================================================== #
# Grey matter Family (i.e., Used after merging, assume already converge(merge) enough) #
# ==================================================================================== #

ALLEN_FAMILY_TYPE = Literal['HB', 'HY', 'TH', 'MB', 'CB', 'CTXpl', 'HPF', 'ISOCORTEX', 'OLF', 'CTXsp']
AllenFamilyType = tuple[Area, ...]

# VS
VS_FAMILY: AllenFamilyType = ('VS',)

# HB
HB_FAMILY: AllenFamilyType = ('MY-mot', 'MY-sat', 'MY-sen', 'P-mot', 'P-sat', 'P-sen')

# HY
HY_FAMILY: AllenFamilyType = ('HY',)

# TH
TH_FAMILY: AllenFamilyType = ('TH',)

# MB (`MB` initially show in ccf map)
MB_FAMILY: AllenFamilyType = ('MB', 'MBmot', 'MBsen', 'MBsta')

# CB
CB_FAMILY: AllenFamilyType = ('CBN', 'CBX')

# CNU
CNU_FAMILY: AllenFamilyType = ('PAL', 'STR')

# CTXpl
HPF_FAMILY: AllenFamilyType = ('HPF',)
ISOCORTEX_FAMILY: AllenFamilyType = (
    'ACA', 'AI', 'AUD', 'ECT', 'FRP', 'GU', 'ILA', 'MO', 'ORB', 'PERI', 'PL', 'PTLp', 'RSP',
    'SS', 'TEa', 'VIS', 'VISC'
)
OLF_FAMILY: AllenFamilyType = ('OLF',)
CTXPL_FAMILY = HPF_FAMILY + ISOCORTEX_FAMILY + OLF_FAMILY

# CTXsp
CTXSP_FAMILY: AllenFamilyType = ('CTXsp', 'BLA', 'BMA', 'CLA', 'EP', 'LA', 'PA')

DEFAULT_FAMILY_DICT: dict[str, AllenFamilyType] = dict(
    VS=VS_FAMILY,
    HB=HB_FAMILY,
    HY=HY_FAMILY,
    TH=TH_FAMILY,
    MB=MB_FAMILY,
    CB=CB_FAMILY,
    CNU=CNU_FAMILY,
    HPF=HPF_FAMILY,
    ISOCORTEX=ISOCORTEX_FAMILY,
    OLF=OLF_FAMILY,
    CTXSP=CTXSP_FAMILY
)


# ============== #
# Implementation #
# ============== #

def merge_area(ps: Series, region: dict[str, list[Area]]) -> list[str]:
    """
    merge the area name series to another level based on the ``region`` dict

    :param ps: pandas/polars series with area name
    :param region: ``MERGE_REGION_LV*``
    :return:
    """
    ret = []
    for a in ps:
        ret.append(_merge_area(a, region))

    return ret


def _merge_area(s: str, region: dict[str, list[Area]]) -> str:
    """

    :param s: individual `acronym` value
    :param region:
    :return:
    """
    for n, p in region.items():
        for pattern in p:
            if pattern.endswith('*'):
                if s.startswith(pattern[:-1]):
                    return n
            elif pattern == s:
                return n

    return s


def merge_until_level(ps: Series, level: TreeLevel) -> list[Area]:
    """
    merge the area until which `level`

    :param ps: pandas/polars series with area name
    :param level: level int
    :return: list of area
    """
    if level not in list(range(NUM_MERGE_LAYER)):
        raise ValueError(f'wrong level: {level}')

    if level <= MERGE_LEVEL_LAYER:
        ps = merge_area(ps, MERGE_REGION_LV4)
    if level <= MERGE_LEVEL_DVMLAP:
        ps = merge_area(ps, MERGE_REGION_LV3)
    if level <= MERGE_LEVEL_C2:
        ps = merge_area(ps, MERGE_REGION_LV2)
    if level <= MERGE_LEVEL_C1:
        ps = merge_area(ps, MERGE_REGION_LV1)
    if level <= MERGE_LEVEL_C0:
        ps = merge_area(ps, MERGE_REGION_LV0)

    return ps
