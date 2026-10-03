from nfl_vision.geometria import caixa_inteira


def test_arredonda_coordenadas():
    assert caixa_inteira((10.4, 20.6, 30.5, 40.49), (100, 200, 3)) == (10, 21, 30, 40)


def test_limita_a_imagem():
    assert caixa_inteira((-5.0, -1.2, 250.0, 130.0), (100, 200, 3)) == (0, 0, 200, 100)


def test_aceita_shape_2d():
    assert caixa_inteira((1, 2, 3, 4), (10, 10)) == (1, 2, 3, 4)
