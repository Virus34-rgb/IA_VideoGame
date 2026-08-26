
import numpy

from sumTree import SumTree



def comprobar_arbol(tree_individual, tree_batch, nombre_prueba):
    print(f"\n--- {nombre_prueba} ---")

    tree_ok = numpy.array_equal(
        tree_individual.tree,
        tree_batch.tree
    )

    data_ok = numpy.array_equal(
        tree_individual.data,
        tree_batch.data
    )

    write_ok = tree_individual.write == tree_batch.write
    size_ok = tree_individual.size == tree_batch.size
    total_ok = numpy.isclose(
        tree_individual.total(),
        tree_batch.total()
    )

    print(f"tree : {'OK' if tree_ok else 'ERROR'}")
    print(f"data : {'OK' if data_ok else 'ERROR'}")
    print(f"write: {'OK' if write_ok else 'ERROR'}")
    print(f"size : {'OK' if size_ok else 'ERROR'}")
    print(f"total: {'OK' if total_ok else 'ERROR'}")

    if tree_ok and data_ok and write_ok and size_ok and total_ok:
        print("RESULTADO: OK")
        return True

    print("RESULTADO: ERROR")

    if not tree_ok:
        diferencias = numpy.where(
            tree_individual.tree != tree_batch.tree
        )[0]

        print(f"Primeras diferencias en tree: {diferencias[:10]}")

        for index in diferencias[:10]:
            print(
                f"  índice {index}: "
                f"individual={tree_individual.tree[index]}, "
                f"batch={tree_batch.tree[index]}"
            )

    return False


def prueba_basica():
    capacidad = 100000
    cantidad = 512

    prioridades = numpy.arange(1, cantidad + 1, dtype=float)
    datos = numpy.array(
        [f"dato_{i}" for i in range(cantidad)],
        dtype=object
    )

    tree_individual = SumTree(capacidad)
    tree_batch = SumTree(capacidad)

    # Inserción tradicional
    for prioridad, dato in zip(prioridades, datos):
        tree_individual.add(prioridad, dato)

    # Inserción en lote
    tree_batch.add_batch(prioridades, datos)

    return comprobar_arbol(
        tree_individual,
        tree_batch,
        "PRUEBA BÁSICA (512 elementos)"
    )


def prueba_multiples_batches():
    capacidad = 100000
    batch_size = 512
    numero_batches = 4

    tree_individual = SumTree(capacidad)
    tree_batch = SumTree(capacidad)

    for batch in range(numero_batches):
        prioridades = numpy.arange(
            1,
            batch_size + 1,
            dtype=float
        ) + batch * batch_size

        datos = numpy.array(
            [
                f"batch_{batch}_dato_{i}"
                for i in range(batch_size)
            ],
            dtype=object
        )

        # Una a una
        for prioridad, dato in zip(prioridades, datos):
            tree_individual.add(prioridad, dato)

        # En lote
        tree_batch.add_batch(prioridades, datos)

    return comprobar_arbol(
        tree_individual,
        tree_batch,
        "PRUEBA MÚLTIPLES BATCHES (4 x 512)"
    )


def prueba_wrap_around():
    # Capacidad pequeña para forzar que write vuelva a 0.
    capacidad = 10

    tree_individual = SumTree(capacidad)
    tree_batch = SumTree(capacidad)

    prioridades_1 = numpy.arange(1, 8, dtype=float)
    datos_1 = numpy.array(
        [f"dato_{i}" for i in range(7)],
        dtype=object
    )

    prioridades_2 = numpy.arange(8, 15, dtype=float)
    datos_2 = numpy.array(
        [f"dato_{i}" for i in range(7, 14)],
        dtype=object
    )

    # Primer lote
    for prioridad, dato in zip(prioridades_1, datos_1):
        tree_individual.add(prioridad, dato)

    tree_batch.add_batch(prioridades_1, datos_1)

    # Segundo lote: aquí se produce wrap-around
    for prioridad, dato in zip(prioridades_2, datos_2):
        tree_individual.add(prioridad, dato)

    tree_batch.add_batch(prioridades_2, datos_2)

    return comprobar_arbol(
        tree_individual,
        tree_batch,
        "PRUEBA WRAP-AROUND (capacidad 10)"
    )


def prueba_prioridades_repetidas():
    capacidad = 100000
    cantidad = 512

    # Prioridades repetidas para comprobar que
    # numpy.add.at() agrupa correctamente los padres.
    prioridades = numpy.array(
        [1, 5, 1, 10, 5, 20, 10, 1] * 64,
        dtype=float
    )

    datos = numpy.array(
        [f"dato_{i}" for i in range(cantidad)],
        dtype=object
    )

    tree_individual = SumTree(capacidad)
    tree_batch = SumTree(capacidad)

    for prioridad, dato in zip(prioridades, datos):
        tree_individual.add(prioridad, dato)

    tree_batch.add_batch(prioridades, datos)

    return comprobar_arbol(
        tree_individual,
        tree_batch,
        "PRUEBA PRIORIDADES REPETIDAS"
    )


def main():
    resultados = [
        prueba_basica(),
        prueba_multiples_batches(),
        prueba_wrap_around(),
        prueba_prioridades_repetidas(),
    ]

    print("\n" + "=" * 50)

    if all(resultados):
        print("TODAS LAS PRUEBAS HAN PASADO")
    else:
        print("ALGUNA PRUEBA HA FALLADO")

    print("=" * 50)


if __name__ == "__main__":
    main()
