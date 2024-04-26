import os
import warnings
import argparse

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.poi_categorization_job import PoiCategorizationJob
from job.matrix_generation_for_poi_categorization_job import MatrixGenerationForPoiCategorizationJob


def execute_poi_categorization():
    print("Você escolheu executar o PGC-NN")
    job = PoiCategorizationJob()
    job.start()


def generate_matrix_for_poi_categorization():
    print("Você escolheu gerar as entradas para o PGC-NN")
    job = MatrixGenerationForPoiCategorizationJob()
    job.start()


def exibir_menu(args):
    if args.execute:
        execute_poi_categorization()
    elif args.generate:
        generate_matrix_for_poi_categorization()
    else:
        opcoes = ["Executar", "Gerar entradas", "Sair"]
        while True:
            print("-" * 27)
            print("|          PGC-NN         |")
            print("-" * 27)

            for i, opcao in enumerate(opcoes, start=1):
                print(f"| {i}. {opcao:<20} |")

            print("-" * 27)

            escolha = input("Escolha uma opção (1-3): ")

            if escolha == "1":
                execute_poi_categorization()
            elif escolha == "2":
                generate_matrix_for_poi_categorization()
            elif escolha == "3":
                print("Saindo do programa!")
                break
            else:
                print("Opção inválida. Tente novamente.")


def main():
    parser = argparse.ArgumentParser(description="PGC-NN Command Line Interface")
    parser.add_argument('--execute', action='store_true', help='Execute the Poi Categorization Job directly')
    parser.add_argument('--generate', action='store_true', help='Generate Matrix for Poi Categorization Job directly')
    args = parser.parse_args()
    exibir_menu(args)


if __name__ == "__main__":
    main()