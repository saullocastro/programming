import sympy
from sympy import Function, Derivative, Expr, Add, Mul, Pow, Subs, Symbol
from sympy import Matrix

class myFunction(Function):
    def __init__(self, *args):
        super(myFunction, self).__init__(*args)

def func2der(expr):
    '''Makes:
        u,xt(u(x, t), x, t) = Derivative(u(x, t), x, t)

    '''
    if expr.is_Matrix:
        if expr.shape == (1, 1):
            expr = expr[0,0]
        else:
            raise NotImplementedError('Matrices or vectors are not supported!')
    def _main(expr):
        _new = []
        for a in expr.args:
            is_V = False
            if isinstance(a, V):
                is_V = True
                a = a.expr
            if a.is_Function:
                name = a.__class__.__name__
                for i in a.args:
                    if i.is_Function:
                        func = i
                        break
                if ',' in name:
                    variables = [eval(i) for i in name.split(',')[1]]
                    a = Derivative(func, *variables)
                if 'V' in name:
                    #TODO remove this V and use class
                    a = V(a)
                    a.function = func
            #TODO add more, maybe all that have args
            elif a.is_Add or a.is_Mul or a.is_Pow:
                a = _main(a)
            if is_V:
                a = V(a)
                a.function = func
            _new.append( a )
        return expr.func(*tuple(_new))
    return _main(expr)

def der2func(expr):
    '''Makes:
        Derivative(u(x, t), x, t) = u,xt(u(x, t), x, t)

    '''
    if expr.is_Matrix:
        if expr.shape == (1, 1):
            expr = expr[0,0]
        else:
            raise NotImplementedError('Matrices or vectors are not supported!')
    def _main(expr):
        _new = []
        for a in expr.args:
            is_V = False
            if isinstance(a, V):
                is_V = True
                a = a.expr
            if a.is_Derivative:
                variables = a.atoms()
                func = a.expr
                variables.add(func)
                name = a.expr.__class__.__name__
                if ',' in name:
                    a = Function('%s' % name +
                          ''.join(map(str, a.variables)))(*variables)
                else:
                    a = Function('%s' % name + ',' +
                          ''.join(map(str, a.variables)))(*variables)
            #TODO add more, maybe all that have args
            elif a.is_Add or a.is_Mul or a.is_Pow:
                a = _main(a)
            if is_V:
                a = V(a)
                a.function = func
            _new.append( a )
        return expr.func(*tuple(_new))
    return _main(expr)

def subs2func(expr):
    '''Makes:
        Subs(Derivative(w,x(a, x, t), a), (a,), (b,)) = Dw,x(b, x, t)

    '''
    def _main(expr):
        _new = []
        for a in expr.args:
            if isinstance(a, Subs):
                f = a.args[0].expr
                args = tuple((i for i in f.args if not i in a.args[1]))
                args += a.args[2]
                #TODO remove this V and use a class
                #TODO add a check if some other function has ^V
                a = Function('V%s' % f.__class__.__name__)(*args)
            #TODO add more, maybe all that have args
            elif a.is_Add or a.is_Mul or a.is_Pow:
                a = _main(a)
            _new.append( a )
        return expr.func(*tuple(_new))
    return _main(expr)

class V(Expr):
    def __init__(self, expr, *args):
        super(V, self).__init__(*args)
        self.function = None
        self.expr = expr

    def __repr__(self):
        return 'V' + repr(self.expr)

    def __str__(self):
        return self.__repr__()

    def _eval_derivative(self, *variables):
        expr = self.expr.diff(*variables)
        v = V(expr)
        v.function = self.function
        return v

class Vexpr(Expr):
    __slots__ = ['functions', '_functions', 'expr',
            'integrands', 'non_integrands']

    def __init__(self, expr, *functions):
        self._functions = []
        self.integrands = {}
        self.non_integrands = {}
        if functions[0]=='NOTEVAL':
            self.expr = expr
            self.functions = functions[1:]
            return
        else:
            self.expr = expr
            self.functions = functions
            self._include_variational_operator()
            self._integrate_by_parts()

    def __repr__(self):
        return self.expr.__repr__()

    def __str__(self):
        return self.__repr__()

    def _eval_derivative(self, *variables):
        expr = func2der(self.expr)
        expr = expr.diff(*variables)
        expr = der2func(expr)
        functions = ('NOTEVAL',) + self.functions
        return Vexpr(expr, *functions)

    def _include_variational_operator(self):
            _add = ()
            expr = der2func(self.expr)
            #expr = func2der(self.expr)
            for function in self.functions:
                name = function.__class__.__name__
                _function = Symbol(name.upper()*3)
                derivate = _function*expr.diff(function)
                self._functions.append(_function)
                _add += (derivate, )
            expr = Add(*_add)
            expr = subs2func(expr)
            self.expr = sympy.simplify(expr)

    def _integrate_by_parts(self):
        '''Integrates by parts changing ``Integral(a*Dw,x, x)`` into:
        ``Integral(-a,x*Dw) + a*Dw``
        The resulting expressions inside and outside the integrand
        are stored in ``self.integrand`` and ``self.non_integrand``,
        respectively.

        '''
        def _aux_integrate_by_parts(a):
            variables = None
            needs_integration_by_parts = False
            new = []
            for i in a.args:
                if isinstance(i, V):
                    if not variables:
                        needs_integration_by_parts = True
                        varexpr = i
                        der = varexpr.expr
                        func = der.expr
                        variables = der.variables
                    else:
                        raise NotImplementedError(
                                'Two variations in the same expression!')
                else:
                    new.append(i)
            # "integrand" with all the terms but the one with the variational
            # operator
            integrand = a.func(*tuple(new))
            non_integrand = 0
            if not needs_integration_by_parts:
                return integrand, non_integrand
            func_vars = list(der.variables)
            for i, var in enumerate(variables):
                func_vars.remove(var)
                if func_vars:
                    new_varexpr = V(Derivative(func, *tuple(func_vars)))
                    new_varexpr.function = varexpr.function
                    non_integrand += (-1)**(i)*integrand*new_varexpr
                else:
                    non_integrand += (-1)**(i)*integrand
                integrand = (-1)*integrand.diff(var)
            return integrand, non_integrand
        self.integrands = {}
        self.non_integrands = {}
        for _function, function in zip(self._functions, self.functions):
            integrands = []
            non_integrands = []
            d = sympy.collect(self.expr.expand(), _function, evaluate=False)
            a = func2der(d[_function])
            if a.is_Add:
                for b in a.args:
                    integrand, non_integrand = _aux_integrate_by_parts(b)
                    integrands.append(integrand)
                    non_integrands.append(non_integrand)
            elif isinstance(a, Expr):
                integrand, non_integrand = _aux_integrate_by_parts(a)
                integrands.append(integrand)
                non_integrands.append(non_integrand)
            else:
                print(a)
                raise ('Check here, something is wrong!')
            name = function.__class__.__name__
            self.integrands[name] = Add(*integrands)
            self.non_integrands[name] = Add(*non_integrands)


def test_simple():
    sympy.var('x, y, r')
    u = Function('u')(x, y)
    w = Function('w')(x, y)
    f = Function('f')(x, y)
    e = (u.diff(x) + 1./2*w.diff(x,x)**2)*f.diff(x,y) \
            + w.diff(x,y)*f.diff(x,x)
    return Vexpr(e, u, w)

def test_cylinder_clpt():
    '''Test case where the functional corresponds to the internal energy of
    a cylinder using the Classical Laminated Plate Theory (CLPT)

    '''
    from sympy import Matrix

    sympy.var('x, y, r')
    sympy.var('B11, B12, B16, B21, B22, B26, B61, B62, B66')
    sympy.var('D11, D12, D16, D21, D22, D26, D61, D62, D66')

    # displacement field
    u = Function('u')(x, y)
    v = Function('v')(x, y)
    w = Function('w')(x, y)
    # stress function
    f = Function('f')(x, y)
    # laminate constitute matrices B represents B*, see Jones (1999)
    B = Matrix([[B11, B12, B16],
                [B21, B22, B26],
                [B61, B62, B66]])
    # D represents D*, see Jones (1999)
    D = Matrix([[D11, D12, D16],
                [D12, D22, D26],
                [D16, D26, D66]])
    # strain-diplacement equations
    e = Matrix([[u.diff(x) + 1./2*w.diff(x)**2],
                [v.diff(y) + 1./r*w + 1./2*w.diff(y)**2],
                [u.diff(y) + v.diff(x) + w.diff(x)*w.diff(y)]])
    k = Matrix([[  -w.diff(x, x)],
                [  -w.diff(y, y)],
                [-2*w.diff(x, y)]])
    # representing the internal forces using the stress function
    N = Matrix([[  f.diff(y, y)],
                [  f.diff(x, x)],
                [ -f.diff(x, y)]])
    functional = N.T*V(e) - N.T*B*V(k) + k.T*D.T*V(k)
    return Vexpr(functional, u, v, w)

if __name__ == '__main__':
    print test_cylinder_clpt().integrands

#TODO
# implement a class that allows N.T*Vexpr(e, u, v, w), for example
#   and any other type of algebraic operations

# Look the email first
# Build a class Variation that will apply the variational
# Build a class FUnctionalIntegrand that will contain the expression
# to be integrated...
