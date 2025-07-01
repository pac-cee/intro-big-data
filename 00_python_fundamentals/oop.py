"""
Comprehensive Python OOP Tutorial - Library Management System
This example covers all major OOP concepts in Python.
"""

# =============================================
# 1. Classes and Objects
# =============================================
class Book:
    """A basic class representing a book in the library."""
    
    # Class variable - shared among all instances
    total_books = 0
    
    def __init__(self, title, author, isbn, available=True):
        """Initialize book attributes."""
        self.title = title           # Instance variable
        self.author = author         # Instance variable
        self.isbn = isbn             # Instance variable
        self.available = available   # Instance variable
        Book.total_books += 1        # Increment class variable
        
    def __str__(self):
        """String representation of the book."""
        return f"'{self.title}' by {self.author}"
    
    def check_out(self):
        """Mark the book as checked out."""
        if self.available:
            self.available = False
            return f"Successfully checked out {self.title}"
        return f"Sorry, {self.title} is not available"
    
    def return_book(self):
        """Mark the book as returned."""
        self.available = True
        return f"Thank you for returning {self.title}"


# =============================================
# 2. Inheritance
# =============================================
class EBook(Book):
    """A specialized book class for electronic books."""
    
    def __init__(self, title, author, isbn, file_format, file_size_mb):
        super().__init__(title, author, isbn)
        self.file_format = file_format
        self.file_size_mb = file_size_mb
        self.downloaded = False
        
    def download(self):
        """Simulate downloading the ebook."""
        if not self.downloaded:
            self.downloaded = True
            return f"Downloading {self.title} ({self.file_size_mb}MB)..."
        return "This book is already downloaded"
    
    # Method overriding
    def __str__(self):
        return f"EBook: '{self.title}' by {self.author} ({self.file_format})"


# =============================================
# 3. Encapsulation
# =============================================
class LibraryMember:
    """Represents a library member with private attributes."""
    
    def __init__(self, member_id, name):
        self.member_id = member_id
        self._name = name  # Protected attribute
        self.__borrowed_books = []  # Private attribute
        
    def borrow_book(self, book):
        """Borrow a book if available."""
        if book.available:
            self.__borrowed_books.append(book)
            book.check_out()
            return f"{self._name} borrowed {book.title}"
        return f"Could not borrow {book.title}"
    
    def return_book(self, book):
        """Return a borrowed book."""
        if book in self.__borrowed_books:
            self.__borrowed_books.remove(book)
            book.return_book()
            return f"{self._name} returned {book.title}"
        return "Book not found in borrowed items"
    
    def get_borrowed_books(self):
        """Get list of borrowed books (encapsulation)."""
        return [book.title for book in self.__borrowed_books]
    
    @property
    def name(self):
        """Getter for name property."""
        return self._name.title()
    
    @name.setter
    def name(self, new_name):
        """Setter for name property with validation."""
        if isinstance(new_name, str) and len(new_name) > 0:
            self._name = new_name
        else:
            raise ValueError("Name must be a non-empty string")


# =============================================
# 4. Polymorphism
# =============================================
class LibraryItem:
    """Base class for all library items."""
    
    def __init__(self, title, item_id):
        self.title = title
        self.item_id = item_id
        self.checked_out = False
        
    def check_out(self):
        """Check out the item."""
        self.checked_out = True
        return f"Checked out: {self.title}"
    
    def return_item(self):
        """Return the item."""
        self.checked_out = False
        return f"Returned: {self.title}"
    
    def get_info(self):
        """Get item information (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement get_info()")


class DVD(LibraryItem):
    """DVD item in the library."""
    
    def __init__(self, title, item_id, director, duration_minutes):
        super().__init__(title, item_id)
        self.director = director
        self.duration_minutes = duration_minutes
        
    def get_info(self):
        """Return DVD information."""
        return (f"DVD: {self.title} (ID: {self.item_id})\n"
                f"Director: {self.director}\n"
                f"Duration: {self.duration_minutes} minutes")


class Magazine(LibraryItem):
    """Magazine item in the library."""
    
    def __init__(self, title, item_id, issue_number, publisher):
        super().__init__(title, item_id)
        self.issue_number = issue_number
        self.publisher = publisher
        
    def get_info(self):
        """Return magazine information."""
        return (f"Magazine: {self.title} (ID: {self.item_id})\n"
                f"Issue: {self.issue_number}\n"
                f"Publisher: {self.publisher}")


# =============================================
# 5. Class Methods and Static Methods
# =============================================
class Library:
    """Represents a library with class and static methods."""
    
    def __init__(self, name):
        self.name = name
        self.books = []
        self.members = []
        
    def add_book(self, book):
        """Add a book to the library."""
        self.books.append(book)
        return f"Added {book.title} to the library"
    
    def add_member(self, member):
        """Add a member to the library."""
        self.members.append(member)
        return f"Added {member.name} as a library member"
    
    @classmethod
    def create_from_books(cls, name, book_list):
        """Create a library with an initial collection of books."""
        library = cls(name)
        for book in book_list:
            library.add_book(book)
        return library
    
    @staticmethod
    def get_library_hours():
        """Return the library's operating hours."""
        return "Monday-Friday: 9am-9pm\nSaturday-Sunday: 10am-6pm"


# =============================================
# 6. Magic Methods (Dunder Methods)
# =============================================
class BookShelf:
    """Demonstrates magic methods in Python."""
    
    def __init__(self, max_books=10):
        self.books = []
        self.max_books = max_books
        
    def add_book(self, book):
        """Add a book to the shelf if there's space."""
        if len(self.books) < self.max_books:
            self.books.append(book)
            return True
        return False
    
    def __len__(self):
        """Return the number of books on the shelf."""
        return len(self.books)
    
    def __getitem__(self, index):
        """Enable indexing (e.g., shelf[0])."""
        return self.books[index]
    
    def __contains__(self, book):
        """Enable 'in' operator (e.g., if book in shelf)."""
        return book in self.books
    
    def __iter__(self):
        """Enable iteration over the shelf."""
        return iter(self.books)
    
    def __str__(self):
        """String representation of the bookshelf."""
        if not self.books:
            return "The bookshelf is empty"
        book_list = "\n".join(f"- {book}" for book in self.books)
        return f"Books on shelf:\n{book_list}"


# =============================================
# 7. Composition
# =============================================
class LibraryBranch:
    """Demonstrates composition - a library has many books."""
    
    def __init__(self, name, location):
        self.name = name
        self.location = location
        self.books = []  # Composition: LibraryBranch has many Books
        
    def add_book(self, book):
        """Add a book to this branch."""
        self.books.append(book)
        return f"Added {book.title} to {self.name} branch"
    
    def get_available_books(self):
        """Get all available books at this branch."""
        return [book for book in self.books if book.available]
    
    def __str__(self):
        return f"{self.name} Library Branch at {self.location}"


# =============================================
# 8. Abstract Base Classes
# =============================================
from abc import ABC, abstractmethod

class Borrowable(ABC):
    """Abstract base class for items that can be borrowed."""
    
    @abstractmethod
    def check_out(self):
        """Check out the item."""
        pass
    
    @abstractmethod
    def return_item(self):
        """Return the item."""
        pass
    
    @abstractmethod
    def is_available(self):
        """Check if the item is available."""
        pass


class AudioBook(Borrowable):
    """An audiobook that implements the Borrowable interface."""
    
    def __init__(self, title, author, narrator, duration_hours):
        self.title = title
        self.author = author
        self.narrator = narrator
        self.duration_hours = duration_hours
        self._checked_out = False
        
    def check_out(self):
        if not self._checked_out:
            self._checked_out = True
            return f"Checked out: {self.title} (Audiobook)"
        return "Already checked out"
    
    def return_item(self):
        if self._checked_out:
            self._checked_out = False
            return f"Returned: {self.title} (Audiobook)"
        return "Wasn't checked out"
    
    def is_available(self):
        return not self._checked_out
    
    def __str__(self):
        return (f"AudioBook: {self.title} by {self.author}\n"
                f"Narrated by: {self.narrator}\n"
                f"Duration: {self.duration_hours} hours")


# =============================================
# Example Usage
# =============================================
def demonstrate_oop_concepts():
    """Demonstrate all OOP concepts with practical examples."""
    print("=" * 50)
    print("PYTHON OOP TUTORIAL - LIBRARY MANAGEMENT SYSTEM")
    print("=" * 50)
    
    # 1. Creating objects and using class variables
    print("\n1. Creating Book objects:")
    book1 = Book("Python Crash Course", "Eric Matthes", "978-1593279288")
    book2 = Book("Fluent Python", "Luciano Ramalho", "978-1491946008")
    print(book1)  # Uses __str__ method
    print(f"Total books in library: {Book.total_books}")
    
    # 2. Inheritance
    print("\n2. Inheritance with EBook:")
    ebook = EBook("Automate the Boring Stuff", "Al Sweigart", 
                  "978-1593275990", "PDF", 10.5)
    print(ebook)
    print(ebook.download())
    print(ebook.download())  # Already downloaded
    
    # 3. Encapsulation
    print("\n3. Encapsulation with LibraryMember:")
    member = LibraryMember("M001", "john doe")
    print(member.borrow_book(book1))
    print(member.borrow_book(book2))  # Try to borrow second book
    print(f"Borrowed books: {member.get_borrowed_books()}")
    member.name = "john smith"  # Using property setter
    print(f"Member's name (using getter): {member.name}")
    
    # 4. Polymorphism
    print("\n4. Polymorphism with LibraryItem subclasses:")
    items = [
        DVD("The Matrix", "DVD001", "The Wachowskis", 136),
        Magazine("National Geographic", "MAG001", 256, "National Geographic Society")
    ]
    for item in items:
        print("\n" + item.get_info())
        print(item.check_out())
        print(item.return_item())
    
    # 5. Class methods and static methods
    print("\n5. Class and Static Methods:")
    central_library = Library.create_from_books("Central Library", [book1, book2])
    print(f"Created {central_library.name} with {len(central_library.books)} books")
    print("Library hours:")
    print(Library.get_library_hours())
    
    # 6. Magic methods
    print("\n6. Magic Methods with BookShelf:")
    shelf = BookShelf(3)
    shelf.add_book(book1)
    shelf.add_book(book2)
    print(f"Books on shelf: {len(shelf)}")
    print(f"First book: {shelf[0]}")
    print(f"Is 'Python Crash Course' on shelf? {'Yes' if book1 in shelf else 'No'}")
    print(shelf)  # Uses __str__
    
    # 7. Composition
    print("\n7. Composition with LibraryBranch:")
    downtown_branch = LibraryBranch("Downtown", "123 Main St")
    downtown_branch.add_book(book1)
    downtown_branch.add_book(book2)
    print(downtown_branch)
    print(f"Available books: {[book.title for book in downtown_branch.get_available_books()]}")
    
    # 8. Abstract Base Classes
    print("\n8. Abstract Base Classes with AudioBook:")
    audiobook = AudioBook("The Hobbit", "J.R.R. Tolkien", "Rob Inglis", 11.1)
    print(audiobook)
    print(audiobook.check_out())
    print(f"Available: {audiobook.is_available()}")
    print(audiobook.return_item())
    
    print("\n" + "="*50)
    print("END OF OOP DEMONSTRATION")
    print("="*50)


if __name__ == "__main__":
    demonstrate_oop_concepts()