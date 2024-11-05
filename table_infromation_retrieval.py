from typing import Any

import torch


class LLM():
    def __init__(self):
        self.counter  = 0

    def encode(self, element:str) -> torch.Tensor:
        self.counter += 1
        return torch.Tensor([self.counter])

class CatalogTable:
    def __init__(self, title: Any):
        self.llm = LLM()
        self.title = title
        self.title_embedding = self.llm.encode(title)
        self.content = []
    
    def extract_possible_list_of_values_from_string(self, possible_list_of_values_as_string: str) -> list[str]:
        possible_list_elements =[
            possible_list_element.strip() for
            possible_list_element in ";".join(possible_list_of_values_as_string.split(",")).split(";")
        ]

        number_of_words = len(possible_list_of_values_as_string.split(" "))
        
        if len(possible_list_elements) > number_of_words / 3:
            return possible_list_elements
        
        return [possible_list_of_values_as_string]

    def extract_value_from_catalog_element(self, catalog_element: Any) -> list[str]:
        if type(catalog_element) is str:
            return self.extract_possible_list_of_values_from_string(catalog_element)
        elif type(catalog_element) is list:
            list_of_values = []
            for _catalog_element in catalog_element:
                if (
                    type(_catalog_element) is int or type(_catalog_element) is float
                ):
                    list_of_values.append(str(_catalog_element))
                else:
                    list_of_values.extend(
                                self.extract_possible_list_of_values_from_string(
                            possible_list_of_values_as_string=_catalog_element
                        )
                    )
            return list_of_values
        elif type(catalog_element) is int or type(catalog_element) is float:
            return list(str(catalog_element))
        else:
            raise RuntimeError(f"Unexpected value for catalog in input: {catalog_element}")

    def add_element(self, identifier: int, element: Any):
        list_of_elements = self.extract_value_from_catalog_element(element)
        for element in list_of_elements:
            self.content.append((
                identifier,
                self.llm.encode(element)
            ))
    
class CatalogRetrievalDatabase():
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.llm = LLM()
        self.id_to_data_entry = {
            i: data_entry for i, data_entry in enumerate(data)
        }
        self.table_title_to_table: dict[str, CatalogTable] = {}
        self.populate_list_of_tables(data)

        self.table_titlte_to_embedding = {
            title : self.llm.encode(title)
            for title, _ in self.table_title_to_table.items()
        }
    
    def populate_list_of_tables(self, data: list[dict[str, Any]]) -> None:
        for id, catalog_entry in self.id_to_data_entry.items():
            for title, value in catalog_entry.items():
                if title in self.table_title_to_table:
                    table = self.table_title_to_table[title]
                    table.add_element(id, value)
                else:
                    self.table_title_to_table[title] = CatalogTable(title)
                    self.table_title_to_table[title].add_element(id, value)


def use_case_1():
    input: list[dict[str, Any]] = [
        {"type_1": [12, 2, 3,"4"]},
        {
         "type_1": [5, 6],
         "type_2": "word in a sentece",
         "type_4": 15.3
        },
    ]

    database = CatalogRetrievalDatabase(input)
    print(database.table_title_to_table["type_1"].content)