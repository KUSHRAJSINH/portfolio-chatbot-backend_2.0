from pipeline import run_pipeline

name=input("enter name: ")
data = run_pipeline(name)

print("\nFINAL RESULT:\n", data)