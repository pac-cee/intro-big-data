# Simple OOP Project: Library System

class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
        self.is_checked_out = False

    def __str__(self):
        status = 'Checked Out' if self.is_checked_out else 'Available'
        return f'"{self.title}" by {self.author} ({self.year}) - {status}'

    def check_out(self):
        if not self.is_checked_out:
            self.is_checked_out = True
            print(f'You checked out {self.title}.')
        else:
            print(f'{self.title} is already checked out.')

    def return_book(self):
        if self.is_checked_out:
            self.is_checked_out = False
            print(f'You returned {self.title}.')
        else:
            print(f'{self.title} was not checked out.')

class Library:
    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        self.books.append(book)
        print(f'Added {book.title} to the library.')

    def list_books(self):
        print(f'Books in {self.name}:')
        for book in self.books:
            print(book)

    def find_book(self, title):
        for book in self.books:
            if book.title.lower() == title.lower():
                return book
        return None

    def check_out_book(self, title):
        book = self.find_book(title)
        if book:
            book.check_out()
        else:
            print(f'Book titled "{title}" not found.')

    def return_book(self, title):
        book = self.find_book(title)
        if book:
            book.return_book()
        else:
            print(f'Book titled "{title}" not found.')

# Example usage:
if __name__ == "__main__":
    library = Library("City Library")
    book1 = Book("1984", "George Orwell", 1949)
    book2 = Book("To Kill a Mockingbird", "Harper Lee", 1960)
    book3 = Book("The Great Gatsby", "F. Scott Fitzgerald", 1925)

    library.add_book(book1)
    library.add_book(book2)
    library.add_book(book3)

    library.list_books()
    library.check_out_book("1984")
    library.list_books()
    library.return_book("1984")
    library.list_books()