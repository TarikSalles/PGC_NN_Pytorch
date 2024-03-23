import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.poi_categorization_job import PoiCategorizationJob
from job.matrix_generation_for_poi_categorization_job import MatrixGenerationForPoiCategorizationJob

def exibir_menu():
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
            print("Você escolheu executar o PGC-NN")
            job = PoiCategorizationJob()
            job.start()
        elif escolha == "2":
            print("Você escolheu gerar as entradas para o PGC-NN")
            job = MatrixGenerationForPoiCategorizationJob()
            job.start()
        elif escolha == "3":
            print("Saindo do programa!")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    exibir_menu()