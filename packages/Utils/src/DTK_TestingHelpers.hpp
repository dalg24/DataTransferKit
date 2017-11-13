#ifndef DTK_TESTING_HELPERS_HPP
#define DTK_TESTING_HELPERS_HPP

#include <exception>
#include <regex>
#include <stdexcept>

#define TEUCHOS_TEST_THROW_REGEX_MATCH( code, pattern, out, success )          \
    {                                                                          \
        std::ostream &l_out = ( out );                                         \
        try                                                                    \
        {                                                                      \
            l_out << "Test that code {" #code ";} throws "                     \
                  << "an exception with a message that matches the pattern "   \
                  << "\"" << pattern << "\": ";                                \
            code;                                                              \
            ( success ) = false;                                               \
            l_out << "failed (code did not throw an exception at all)\n";      \
        }                                                                      \
        catch ( std::exception & except )                                      \
        {                                                                      \
            if ( std::regex_match( except.what(), std::regex( pattern ) ) )    \
            {                                                                  \
                l_out << "passed\n";                                           \
                l_out << "\nException message for expected exception:\n\n";    \
                {                                                              \
                    Teuchos::OSTab l_tab( out );                               \
                    l_out << except.what() << "\n\n";                          \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                l_out << "The code was supposed to throw an exception"         \
                      << "with a message that matches the pattern "            \
                      << "\"" << pattern << "\""                               \
                      << ", but it did not match the regular expression."      \
                      << "The exception's message is:\n\n";                    \
                {                                                              \
                    Teuchos::OSTab l_tab( out );                               \
                    l_out << except.what() << "\n\n";                          \
                }                                                              \
                ( success ) = false;                                           \
                l_out << "failed\n";                                           \
            }                                                                  \
        }                                                                      \
        catch ( ... )                                                          \
        {                                                                      \
            l_out                                                              \
                << "The code was supposed to throw an exception that derives " \
                << "from std::exception and with a message that matches the "  \
                << "pattern \"" << pattern << "\", but "                       \
                << "instead threw an exception of some unknown type, which is" \
                << "not a subclass of std::exception.  This means we cannot "  \
                << "show you the exception's message, if it even has "         \
                   "one.\n\n";                                                 \
            ( success ) = false;                                               \
            l_out << "failed\n";                                               \
        }                                                                      \
    }

#define TEST_THROW_REGEX_MATCH( code, regex )                                  \
    TEUCHOS_TEST_THROW_REGEX_MATCH( code, regex, out, success )

#endif
