#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>

namespace Json {

class Value {
private:
    enum Type {
        NULL_VALUE,
        BOOL_VALUE,
        INT_VALUE,
        UINT_VALUE,
        UINT64_VALUE,
        DOUBLE_VALUE,
        STRING_VALUE,
        ARRAY_VALUE,
        OBJECT_VALUE
    };

    Type type_;
    union {
        bool bool_value_;
        int int_value_;
        unsigned int uint_value_;
        uint64_t uint64_value_;
        double double_value_;
    };
    std::string string_value_;
    std::vector<Value> array_value_;
    std::map<std::string, Value> object_value_;

public:
    Value() : type_(NULL_VALUE) {}
    Value(bool value) : type_(BOOL_VALUE), bool_value_(value) {}
    Value(int value) : type_(INT_VALUE), int_value_(value) {}
    Value(unsigned int value) : type_(UINT_VALUE), uint_value_(value) {}
    Value(uint64_t value) : type_(UINT64_VALUE), uint64_value_(value) {}
    Value(double value) : type_(DOUBLE_VALUE), double_value_(value) {}
    Value(const char* value) : type_(STRING_VALUE), string_value_(value) {}
    Value(const std::string& value) : type_(STRING_VALUE), string_value_(value) {}

    bool isNull() const { return type_ == NULL_VALUE; }
    bool isBool() const { return type_ == BOOL_VALUE; }
    bool isInt() const { return type_ == INT_VALUE; }
    bool isUInt() const { return type_ == UINT_VALUE; }
    bool isUInt64() const { return type_ == UINT64_VALUE; }
    bool isDouble() const { return type_ == DOUBLE_VALUE; }
    bool isString() const { return type_ == STRING_VALUE; }
    bool isArray() const { return type_ == ARRAY_VALUE; }
    bool isObject() const { return type_ == OBJECT_VALUE; }

    bool asBool() const {
        if (type_ == BOOL_VALUE) return bool_value_;
        if (type_ == INT_VALUE) return int_value_ != 0;
        return false;
    }

    int asInt() const {
        if (type_ == INT_VALUE) return int_value_;
        if (type_ == UINT_VALUE) return static_cast<int>(uint_value_);
        if (type_ == BOOL_VALUE) return bool_value_ ? 1 : 0;
        return 0;
    }

    unsigned int asUInt() const {
        if (type_ == UINT_VALUE) return uint_value_;
        if (type_ == INT_VALUE) return static_cast<unsigned int>(int_value_);
        return 0;
    }

    uint64_t asUInt64() const {
        if (type_ == UINT64_VALUE) return uint64_value_;
        if (type_ == UINT_VALUE) return static_cast<uint64_t>(uint_value_);
        if (type_ == INT_VALUE) return static_cast<uint64_t>(int_value_);
        return 0;
    }

    double asDouble() const {
        if (type_ == DOUBLE_VALUE) return double_value_;
        if (type_ == INT_VALUE) return static_cast<double>(int_value_);
        return 0.0;
    }

    std::string asString() const {
        if (type_ == STRING_VALUE) return string_value_;
        return "";
    }

    bool isMember(const std::string& key) const {
        if (type_ != OBJECT_VALUE) return false;
        return object_value_.find(key) != object_value_.end();
    }

    Value& operator[](const std::string& key) {
        if (type_ != OBJECT_VALUE) {
            type_ = OBJECT_VALUE;
            object_value_.clear();
        }
        return object_value_[key];
    }

    const Value& operator[](const std::string& key) const {
        static Value null_value;
        if (type_ != OBJECT_VALUE) return null_value;
        auto it = object_value_.find(key);
        return (it != object_value_.end()) ? it->second : null_value;
    }

    Value& operator[](int index) {
        if (type_ != ARRAY_VALUE) {
            type_ = ARRAY_VALUE;
            array_value_.clear();
        }
        if (index >= static_cast<int>(array_value_.size())) {
            array_value_.resize(index + 1);
        }
        return array_value_[index];
    }

    const Value& operator[](int index) const {
        static Value null_value;
        if (type_ != ARRAY_VALUE || index < 0 || index >= static_cast<int>(array_value_.size())) {
            return null_value;
        }
        return array_value_[index];
    }

    void append(const Value& value) {
        if (type_ != ARRAY_VALUE) {
            type_ = ARRAY_VALUE;
            array_value_.clear();
        }
        array_value_.push_back(value);
    }

    size_t size() const {
        if (type_ == ARRAY_VALUE) return array_value_.size();
        if (type_ == OBJECT_VALUE) return object_value_.size();
        return 0;
    }
};

class Reader {
private:
    std::string errors_;

    void skipWhitespace(const std::string& json, size_t& pos) {
        while (pos < json.length() && std::isspace(json[pos])) {
            pos++;
        }
    }

    Value parseValue(const std::string& json, size_t& pos) {
        skipWhitespace(json, pos);

        if (pos >= json.length()) {
            errors_ = "Unexpected end of input";
            return Value();
        }

        char c = json[pos];

        if (c == '{') {
            return parseObject(json, pos);
        } else if (c == '[') {
            return parseArray(json, pos);
        } else if (c == '"') {
            return parseString(json, pos);
        } else if (c == 't' || c == 'f') {
            return parseBool(json, pos);
        } else if (c == 'n') {
            return parseNull(json, pos);
        } else if (std::isdigit(c) || c == '-') {
            return parseNumber(json, pos);
        }

        errors_ = "Unexpected character";
        return Value();
    }

    Value parseObject(const std::string& json, size_t& pos) {
        Value obj;
        pos++; // Skip '{'

        skipWhitespace(json, pos);
        if (pos < json.length() && json[pos] == '}') {
            pos++;
            return obj;
        }

        while (pos < json.length()) {
            skipWhitespace(json, pos);

            // Parse key
            if (json[pos] != '"') {
                errors_ = "Expected string key";
                return Value();
            }

            Value key = parseString(json, pos);
            if (!key.isString()) return Value();

            skipWhitespace(json, pos);
            if (pos >= json.length() || json[pos] != ':') {
                errors_ = "Expected ':'";
                return Value();
            }
            pos++; // Skip ':'

            // Parse value
            Value value = parseValue(json, pos);
            obj[key.asString()] = value;

            skipWhitespace(json, pos);
            if (pos >= json.length()) break;

            if (json[pos] == '}') {
                pos++;
                break;
            } else if (json[pos] == ',') {
                pos++;
            } else {
                errors_ = "Expected ',' or '}'";
                return Value();
            }
        }

        return obj;
    }

    Value parseArray(const std::string& json, size_t& pos) {
        Value arr;
        pos++; // Skip '['

        skipWhitespace(json, pos);
        if (pos < json.length() && json[pos] == ']') {
            pos++;
            return arr;
        }

        while (pos < json.length()) {
            Value value = parseValue(json, pos);
            arr.append(value);

            skipWhitespace(json, pos);
            if (pos >= json.length()) break;

            if (json[pos] == ']') {
                pos++;
                break;
            } else if (json[pos] == ',') {
                pos++;
            } else {
                errors_ = "Expected ',' or ']'";
                return Value();
            }
        }

        return arr;
    }

    Value parseString(const std::string& json, size_t& pos) {
        pos++; // Skip opening quote
        std::string result;

        while (pos < json.length() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.length()) {
                pos++;
                switch (json[pos]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }

        if (pos < json.length()) pos++; // Skip closing quote
        return Value(result);
    }

    Value parseBool(const std::string& json, size_t& pos) {
        if (json.substr(pos, 4) == "true") {
            pos += 4;
            return Value(true);
        } else if (json.substr(pos, 5) == "false") {
            pos += 5;
            return Value(false);
        }
        errors_ = "Invalid boolean value";
        return Value();
    }

    Value parseNull(const std::string& json, size_t& pos) {
        if (json.substr(pos, 4) == "null") {
            pos += 4;
            return Value();
        }
        errors_ = "Invalid null value";
        return Value();
    }

    Value parseNumber(const std::string& json, size_t& pos) {
        size_t start = pos;
        bool isDouble = false;

        if (json[pos] == '-') pos++;

        while (pos < json.length() && std::isdigit(json[pos])) pos++;

        if (pos < json.length() && json[pos] == '.') {
            isDouble = true;
            pos++;
            while (pos < json.length() && std::isdigit(json[pos])) pos++;
        }

        if (pos < json.length() && (json[pos] == 'e' || json[pos] == 'E')) {
            isDouble = true;
            pos++;
            if (pos < json.length() && (json[pos] == '+' || json[pos] == '-')) pos++;
            while (pos < json.length() && std::isdigit(json[pos])) pos++;
        }

        std::string numStr = json.substr(start, pos - start);

        if (isDouble) {
            return Value(std::stod(numStr));
        } else {
            long long num = std::stoll(numStr);
            if (num >= 0 && num <= UINT64_MAX) {
                return Value(static_cast<uint64_t>(num));
            } else {
                return Value(static_cast<int>(num));
            }
        }
    }

public:
    bool parse(const std::string& json, Value& root) {
        errors_.clear();
        size_t pos = 0;
        root = parseValue(json, pos);
        return errors_.empty();
    }

    std::string getFormattedErrorMessages() const {
        return errors_;
    }
};

} // namespace Json
