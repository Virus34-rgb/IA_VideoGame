import constants

class EloRating:
    @staticmethod
    def expected_score(eloA,eloB):
        e = 1 / (1 + 10 ** ((eloB - eloA) / constants.ESTANDAR_ELO) )
        return e
    
    @staticmethod
    def update_elo(elo,expected,result):
        new_elo = elo + constants.K_FACTOR_ELO * (result - expected)
        return float(new_elo)