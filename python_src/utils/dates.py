import calendar
import pandas as pd
import holidays
from datetime import timedelta, datetime

class DateUtils():

    @staticmethod
    def is_date( str_date: str ) -> bool:
        timestamp = pd.to_datetime(str_date, errors='coerce')
        return not pd.isnull( timestamp )

    @staticmethod
    def parse_date( str_date: str ) -> datetime:
        if not str_date:
            raise ValueError(f'parse_date called with empty string')
        timestamp = pd.to_datetime(str_date, errors='coerce', utc=True)
        if pd.isnull( timestamp ):
            raise ValueError(f'parse_date called with invalid date: {str_date}')
        return timestamp  # type: ignore

    @staticmethod
    def next_business_date(date: datetime) -> datetime:
        if date.weekday() == 6: # Sunday
            result = date + timedelta(days=1) # type: ignore
        elif date.weekday() == 5: # Saturday
            result = date + timedelta(days=2) # type: ignore
        else:
            result = date
        nyse_holidays = holidays.financial_holidays('NYSE')
        while(result in nyse_holidays):
            print(f'pushing next_business_day {result} due to NYSE holiday.')
            result = result + timedelta(days=1)
        if date.weekday() == 5:
            return DateUtils.next_business_date(result)  # One recursion in case NYSE holiday falls on a Friday
        else:
            return result

    @staticmethod
    def sort_month_names(months: list[str]) -> list[str]:
        month_number_lookup = {name.lower(): num for num, name in enumerate(calendar.month_name) if num}
        return sorted(months, key=lambda x: month_number_lookup[x.lower()])

    @staticmethod
    def format_date(date: datetime) -> str:
        return date.strftime('%B %d, %Y')

    @staticmethod
    def date_to_iso_string(date: datetime) -> str:
        return date.strftime('%Y-%m-%d')

    @staticmethod
    def date_from_iso_string(iso_date: str) -> datetime:
        return datetime.fromisoformat(iso_date)

    @staticmethod
    def format_datetime(date: datetime) -> str:
        return date.strftime('%B %d, %Y %H:%M:%S')

    @staticmethod
    def datetime_to_iso_string(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def datetime_from_iso_string(iso_datetime: str) -> datetime:
        return datetime.fromisoformat(iso_datetime)
