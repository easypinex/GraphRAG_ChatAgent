import json

class ModelSerialization:
    def to_dict(self):
        result = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        return result

    def load_from_json(self, json_str):
        data = json.loads(json_str)
        return self.load_from_dict(data)
    
    def load_from_dict (self, data: dict):
        for column in self.__table__.columns:
            if column.name in data:
                setattr(self, column.name, data[column.name])
        return self