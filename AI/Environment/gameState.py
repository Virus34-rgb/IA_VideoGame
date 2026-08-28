class GameState:
    """Estado inmutable de una partida (para observación)."""
    def __init__(
        self,
        p1_disposition,
        p2_disposition,
        p1_deaths,
        p2_deaths,
        turn,
    ):
        self.p1_disposition = p1_disposition
        self.p2_disposition = p2_disposition
        self.p1_deaths = p1_deaths
        self.p2_deaths = p2_deaths
        self.turn = turn