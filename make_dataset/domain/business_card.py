from dataclasses import dataclass


@dataclass
class BusinessCard:
    image_path: str
    prompt: str
    company: str
    name: str
    email: str
    phone_number: str
    address: str

    def create_assistant(self) -> str:
        return (
            "{"+f'"Company":"{self.company}","Name":"{self.name}","Email":"{self.email}","Phone Number":"{self.phone_number}","Address":"{self.address}"'+"}"
        )

    def create_conversations(self) -> dict:
        return {
            "id":0,
            "image":self.image_path,
            "conversations":[
            {
                "from": "human",
                "value": self.prompt,
            },
            {
                "from": "gpt",
                "value": self.create_assistant(),
            },
        ]
        }

    @property
    def xml(self) -> str:
        return (
            "<s>"
            f"<s_company>{self.company}</s_company>"
            f"<s_name>{self.name}</s_name>"
            f"<s_email>{self.email}</s_email>"
            f"<s_phone_number>{self.phone_number}</s_phone_number>"
            f"<s_address>{self.address}</s_address>"
            "</s>"
        )

    @classmethod
    def get_xml_tags(cls) -> list[str]:
        return [
            "<s>",
            "<s_company>",
            "</s_company>",
            "<s_name>",
            "</s_name>",
            "<s_email>",
            "</s_email>",
            "<s_phone_number>",
            "</s_phone_number>",
            "<s_address>",
            "</s_address>",
            "</s>",
        ]
