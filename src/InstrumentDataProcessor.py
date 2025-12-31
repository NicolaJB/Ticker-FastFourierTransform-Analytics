# import library modules
import csv
import datetime

# processes and saves data to output CSV from files data.txt, DynamicField.txt and StaticFields.txt
class InstrumentDataProcessor:
    def __init__(self, data_file, static_fields_file, dynamic_fields_file):  # parses in original file contents
        self.data_file = data_file  # initialise class with file names
        self.static_fields_file = static_fields_file
        self.dynamic_fields_file = dynamic_fields_file
        self.instrument_codes = []  # sets up empty lists/dictionaries to store relevant file data
        self.static_fields = {}
        self.dynamic_fields = {}

    def extract_instrument_codes(self):  # extracts list of instrument codes from data file (data.txt)
        with open(self.data_file, 'r') as file:
            for line in file:
                fields = line.strip().split('|')
                if len(fields) >= 4:
                    self.instrument_codes.append(fields[3])  # stores codes in instrument_codes list

    def extract_timestamps(self):  # extracts start/end timestamps from data file (data.txt)
        start_timestamp = None
        end_timestamp = None
        with open(self.data_file, 'r') as file:
            lines = file.readlines()
            if lines:
                start_timestamp = lines[0].split('|')[1]
                end_timestamp = lines[-1].split('|')[1]
        return start_timestamp, end_timestamp

    def extract_date(self):  # extracts data logging date from first record in data file (data.txt)
        with open(self.data_file, 'r') as file:
            first_line = file.readline()
            if first_line:
                date_str = first_line.split('|')[0]
                return date_str.replace('-', '')
        return None

    def load_fields(self, filename, prefix):  # loads static/dynamic field data from StaticFields.txt/DynamicField.txt
        fields = {}
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    field_id, description = parts  # find field ID before tab, description after tab in each row
                    if field_id.startswith(prefix):
                        fields[field_id[1:]] = description  # stores S/D key:ID and value:description in dictionary
        return fields

    def parse_data_file(self, start_timestamp, end_timestamp):
        field_list = []
        with open(self.data_file, 'r') as file:  # open data.txt file
            for line in file:
                parts = line.strip().split('|')
                if len(parts) < 8:  # checks items up until 8th item in row
                    continue

                timestamp = parts[1]  # 2nd item
                instrument_code = parts[3]  # 4th item
                field_type = parts[2]  # third item
                fields_data = parts[7:]  # from 8th item (where fields 'f=' begin)

                if instrument_code in self.instrument_codes and start_timestamp <= timestamp <= end_timestamp:
                    for field in fields_data:  # from 8th item onwards, find ID after 'f' and value after '='
                        if field.startswith('f'):
                            field_parts = field[1:].split('=')
                            if len(field_parts) == 2:
                                field_id, value = field_parts
                                if field_type == 'S' and field_id in self.static_fields:  # match 'f' ID with S 'ID'
                                    # append S description with matched f value/details to field_list for return
                                    field_list.append(
                                        (instrument_code, timestamp, field_id, self.static_fields[field_id], value))
                                elif field_type != 'S' and field_id in self.dynamic_fields:  # match 'f' ID with D 'ID'
                                    # append D description with matched f value/details to field_list for return
                                    field_list.append(
                                        (instrument_code, timestamp, field_id, self.dynamic_fields[field_id], value))
        return field_list

    def save_to_csv(self, field_mappings, output_filename):  # prepare to save field_list as field_mappings to CSV
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Instrument Code", "Timestamp", "Field ID", "Description", "Value"])
            writer.writerows(field_mappings)

    def process(self):  # program's main process - to extract, parse, and save data
        self.extract_instrument_codes()
        start_timestamp, end_timestamp = self.extract_timestamps()  # extract timestamps
        self.static_fields = self.load_fields(self.static_fields_file, "S")  # extract S ID matched field values (parse S prefix)
        self.dynamic_fields = self.load_fields(self.dynamic_fields_file, "D")  # extract D ID matched field values (parse D prefix)
        field_mappings = self.parse_data_file(start_timestamp, end_timestamp)  # assign returned field_list to field_mappings

        date_str = self.extract_date()
        if date_str:
            output_filename = f"output_{date_str}.csv"  # generate filename using date from 1st data.txt record
        else:
            output_filename = "output.csv"  # in case date missing from first row of data.txt

        self.save_to_csv(field_mappings, output_filename)  # save field_list (now field_mappings) to CSV file
        print(f"Output saved to {output_filename}")
        return output_filename


class InstrumentDataSearcher:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename

    def search_instrument_code(self):  # search instrument codes and display corresponding values from output CSV
        while True:
            instrument_code = input("Enter the instrument code to search (or 'exit' to quit): ").strip()
            if instrument_code.lower() == 'exit':
                print("Exiting the search.")
                break

            found = False
            with open(self.csv_filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    if row["Instrument Code"] == instrument_code:
                        if not found:
                            print(f"Instrument Code: {instrument_code} found. Here are the details:")
                            found = True
                        print(f"{row['Timestamp']} - {row['Description']}: {row['Value']}")

            if not found:
                print(f"Instrument Code: {instrument_code} not found in {self.csv_filename}.")


if __name__ == "__main__":  # create instance of InstrumentDataProcessor
    data_processor = InstrumentDataProcessor("data.txt", "StaticFields.txt", "DynamicFields.txt")
    output_file = data_processor.process()  # process the data from the files parsed into the instance

    searcher = InstrumentDataSearcher(output_file)  # create an instance of InstrumentDataSearcher tool
    searcher.search_instrument_code()  # search for inputted instrument code in output CSV
